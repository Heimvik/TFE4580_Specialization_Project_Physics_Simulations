from discretize import CylindricalMesh
from discretize.utils import mkvc

from simpeg import maps
import simpeg.electromagnetics.time_domain as tdem

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from SIMPEG.pi_config import PiConfig
from SIMPEG.pi_plotter import PiPlotter


class PiSimulator:
    def __init__(self, config, loop_z_start=0.2, loop_z_increment=0.05, num_increments=0):
        self.cfg = config
        self.num_increments = num_increments
        self.loop_z_start = float(loop_z_start)
        self.loop_z_increment = float(loop_z_increment)

    def create_survey(self, time_channels, waveform, loop_z):
        # Transmitter location
        xtx, ytx, ztx = np.meshgrid([0], [0], [loop_z])
        source_locations = np.c_[mkvc(xtx), mkvc(ytx), mkvc(ztx)]
        ntx = np.size(xtx)
        # Receiver location
        xrx, yrx, zrx = np.meshgrid([0], [0], [loop_z])
        receiver_locations = np.c_[mkvc(xrx), mkvc(yrx), mkvc(zrx)]

        source_list = []

        for ii in range(ntx):
            dbzdt_receiver = tdem.receivers.PointMagneticFluxTimeDerivative(
                receiver_locations[ii, :], time_channels, "z"
            )
            receivers_list = [dbzdt_receiver]

            source_list.append(
                tdem.sources.CircularLoop(
                    receivers_list,
                    location=source_locations[ii],
                    waveform=waveform,
                    current=self.cfg.tx_current,
                    radius=self.cfg.tx_radius,
                    n_turns=self.cfg.tx_n_turns
                )
            )
        return tdem.Survey(source_list)

    def create_conductivity_model(self, mesh, target_present):
        active_area_z = self.cfg.target_z + self.cfg.target_height/2
        ind_active = mesh.cell_centers[:, 2] < active_area_z

        model_map = maps.InjectActiveCells(mesh, ind_active, self.cfg.air_conductivity)

        r = mesh.cell_centers[ind_active, 0]
        z = mesh.cell_centers[ind_active, 2]

        model = self.cfg.air_conductivity * np.ones(ind_active.sum())

        ind_soil = (
            (z < self.cfg.target_z - self.cfg.target_height/2)
        )
        model[ind_soil] = self.cfg.soil_conductivity

        if target_present:
            ind_can = (
                (r <= self.cfg.target_radius) &
                (z < self.cfg.target_z + self.cfg.target_height/2) &
                (z > self.cfg.target_z - self.cfg.target_height/2)
            )
            model[ind_can] = self.cfg.aluminum_conductivity

        return {"model": model, "model_map": model_map}

    def run(self):
        print("Running simulation with the following configuration:")
        print(self.cfg.summary())

        # 1. Define the waveform to be used, in this case - always a step-off
        waveform = tdem.sources.StepOffWaveform(off_time=self.cfg.waveform_off_time)
        time_channels = np.linspace(0, 1024e-6, 1024)

        # 2. Create the survey
        surveys = []
        for i in range(self.num_increments+2):
            loop_z = self.loop_z_start + i * self.loop_z_increment
            surveys.append(self.create_survey(time_channels, waveform, loop_z))

        # 3. Create the mesh
        hr = [(0.01, 15), (0.01, 15, 1.3), (0.05, 10, 1.5)]
        hphi = 1
        hz = [(0.01, 10, -1.3), (0.01, 30), (0.01, 10, 1.3)]
        mesh = CylindricalMesh([hr, hphi, hz], x0="00C")

        # 4. Create all the different conductivity models
        models = []
        models.append(self.create_conductivity_model(mesh, target_present=False))
        models.append(self.create_conductivity_model(mesh, target_present=True))

        # 5. Create the simulation objects
        simulations = []
        
        # Simulation for baseline without the target
        simulations.append(
            tdem.simulation.Simulation3DMagneticFluxDensity(
                mesh, survey=surveys[0], sigmaMap=models[0]["model_map"], t0=self.cfg.simulation_t0
            )
        )
        simulations[0].time_steps = self.cfg.time_steps

        for i in range(0,self.num_increments+1):
            # Simulation for target present
            simulations.append(
                tdem.simulation.Simulation3DMagneticFluxDensity(
                    mesh, survey=surveys[i], sigmaMap=models[1]["model_map"], t0=self.cfg.simulation_t0
                )
            )
            simulations[i+1].time_steps = self.cfg.time_steps

        decays, times = [],[]
        for i in range(len(simulations)):
            times.append(time_channels)
            if i == 0:
                dpred = simulations[i].dpred(m=models[0]["model"])
            else:
                dpred = simulations[i].dpred(m=models[1]["model"])
            dpred = np.reshape(dpred, (1, len(time_channels)))
            decays.append(-dpred[0, :])
        
        # Prepare plotting info
        plotting_info = {
            'mesh': mesh,
            'model_with_target': models[1]["model"],
            'model_no_target': models[0]["model"],
            'ind_active': mesh.cell_centers[:, 2] < (self.cfg.target_z + self.cfg.target_height/2),
            'cfg': self.cfg,
            'loop_z_start': self.loop_z_start,
            'loop_z_increment': self.loop_z_increment,
            'num_increments': self.num_increments
        }
        
        return times, decays, plotting_info

class PiConditioner:
    def __init__(self):
        pass

    def amplify(self, time, data, time_gain):
        time_gain = np.array(time_gain)
        if time_gain.size > 0:
            time_gain = time_gain[np.argsort(time_gain[:, 0])]
        gain = np.ones_like(time)
        for t, g in time_gain:
            mask = time >= t
            gain[mask] = g
        return data * gain

    def normalize(self, data, max=None):
        max = np.max(np.abs(data)) if max is None else max
        return data/max

    def quantize(self, data, depth, dtype):
        return (np.round(data * 2**depth)).astype(dtype)

cfg = PiConfig('config.json')
simulator = PiSimulator(cfg, loop_z_start=cfg.tx_z, loop_z_increment=0.08, num_increments=5)
conditioner = PiConditioner()

time, data, plotting_info = simulator.run()

plotter = PiPlotter(
    plotting_info,
    plot_linear=True,      # TDEM response linear scale
    plot_loglog=True,      # TDEM response log-log scale
    plot_side_view=True,   # Environment side view (X-Z plane)
    plot_top_view=True,    # Environment top view (X-Y plane)
    plot_3d=True           # 3D environment model
)

plotter.update_times_data(time, data)
plotter.show_plots()

data = conditioner.amplify(time, data, time_gain=[[1e-5,10], [1e-4,100]])
data = conditioner.normalize(data,0.7)
data = conditioner.quantize(data, depth=8, dtype=np.uint8)

plotter.update_times_data(time, data)
plotter.show_plots()