import json
import os
import numpy as np
from typing import Dict, Any, Optional

class PiemsolConfig:
    def __init__(self, config_path: str = 'config.json'):
        self.config_path = config_path
        self._load_config()
        self._set_defaults()
    
    def _load_config(self):
        if os.path.exists(self.config_path):
            print(f"Loading configuration from {self.config_path}...")
            with open(self.config_path, 'r') as f:
                self._config_data = json.load(f)
            print("Configuration loaded successfully.\n")
        else:
            print(f"Warning: Config file '{self.config_path}' not found. Using default parameters.\n")
            self._config_data = {}
    
    def _set_defaults(self):
        self.tx_num = self._config_data.get('tx_num', 1)
        self.tx_x = self._ensure_array(self._config_data.get('tx_x', 0.0))
        self.tx_y = self._ensure_array(self._config_data.get('tx_y', 0.0))
        self.tx_z = self._ensure_array(self._config_data.get('tx_z', 0.3))
        self.tx_current = self._config_data.get('tx_current', 20.0)
        self.tx_radius = self._config_data.get('tx_radius', 0.4)
        self.tx_n_turns = self._config_data.get('tx_n_turns', 30)
        
        self.rx_num = self._config_data.get('rx_num', 1)
        self.rx_x = self._ensure_array(self._config_data.get('rx_x', 0.0))
        self.rx_y = self._ensure_array(self._config_data.get('rx_y', 0.0))
        self.rx_z = self._ensure_array(self._config_data.get('rx_z', 0.3))
        self.rx_radius = self._config_data.get('rx_radius', 0.4)
        self.rx_n_turns = self._config_data.get('rx_n_turns', 30)
        
        self.separation_z = self._config_data.get('separation_z', 0.3)
        
        self.target_x = self._config_data.get('target_x', 0.0)
        self.target_y = self._config_data.get('target_y', 0.0)
        self.target_z = self._config_data.get('target_z', -0.1)
        self.target_center = np.array([self.target_x, self.target_y, self.target_z])
        
        self.target_radius = self._config_data.get('target_radius', 0.0325)
        self.target_height = self._config_data.get('target_height', 0.12)
        self.target_thickness = self._config_data.get('target_thickness', 0.002)

        self.loop_z_range = self._config_data.get('loop_z_range', [0.3, 0.6])
        self.target_z_range = self._config_data.get('target_z_range', [0, 0.3])
        
        self.air_conductivity = self._config_data.get('air_conductivity', 1e-8)
        self.soil_conductivity = self._config_data.get('soil_conductivity', 0.4)
        self.grass_conductivity = self._config_data.get('grass_conductivity', 0.02)
        self.aluminum_conductivity = self._config_data.get('aluminum_conductivity', 3.5e7)
        self.target_conductivity = self._config_data.get('target_conductivity', self.aluminum_conductivity)
        
        self.grass_x_min = self._config_data.get('grass_x_min', -0.3)
        self.grass_x_max = self._config_data.get('grass_x_max', 0.6)
        self.grass_z_min = self._config_data.get('grass_z_min', -0.3)
        self.grass_z_max = self._config_data.get('grass_z_max', 0.0)
        
        self.time_channel_start = self._config_data.get('time_channel_start', 0.0)
        self.time_channel_end = self._config_data.get('time_channel_end', 500e-6)
        self.time_channel_step = self._config_data.get('time_channel_step', 2e-6)
        
        time_steps_raw = self._config_data.get('time_steps', [[1e-6, 100], [1e-5, 400]])
        self.time_steps = [tuple(step) for step in time_steps_raw]
        
        self.mesh_dh = self._config_data.get('mesh_dh', 0.05)
        self.mesh_dom_width = self._config_data.get('mesh_dom_width', 2.0)
        
        self.mesh_topo_refinement = self._config_data.get('mesh_topo_refinement', [0, 0, 0, 1])
        self.mesh_rx_refinement = self._config_data.get('mesh_rx_refinement', [2, 4])
        self.mesh_target_refinement = self._config_data.get('mesh_target_refinement', [0, 2, 4])
        self.mesh_grass_refinement = self._config_data.get('mesh_grass_refinement', [0, 1, 2])
        
        self.target_refinement_box_size = self._config_data.get('target_refinement_box_size', 0.1)
        
        self.waveform_type = self._config_data.get('waveform_type', 'step_off')
        self.waveform_off_time = self._config_data.get('waveform_off_time', 0.0)
        
        self.simulation_t0 = self._config_data.get('simulation_t0', 0.0)
        
    def _ensure_array(self, value):
        if isinstance(value, (list, tuple)):
            return np.array(value)
        else:
            return np.array([value])
    
    def get_time_channels(self) -> np.ndarray:
        return np.arange(
            self.time_channel_start,
            self.time_channel_end,
            self.time_channel_step
        )
    
    def get_target_refinement_box(self) -> np.ndarray:
        half_size = self.target_refinement_box_size
        box = np.array([
            [self.target_center[0] - half_size, 
             self.target_center[1] - half_size, 
             self.target_center[2] - half_size - 0.05],
            [self.target_center[0] + half_size, 
             self.target_center[1] + half_size, 
             self.target_center[2] + half_size + 0.05]
        ])
        return box
    
    def get_grass_box(self) -> np.ndarray:
        box = np.array([
            [self.grass_x_min, -1.0, self.grass_z_min],
            [self.grass_x_max, 1.0, self.grass_z_max]
        ])
        return box
    
    def summary(self) -> str:
        lines = [
            "=" * 60,
            "TDEM Simulation Configuration Summary",
            "=" * 60,
            "",
            "Transmitter Configuration:",
            f"  Number of transmitters: {self.tx_num}",
            f"  Position (x, y, z): ({self.tx_x}, {self.tx_y}, {self.tx_z}) m",
            f"  Current: {self.tx_current} A",
            f"  Radius: {self.tx_radius} m",
            f"  Turns: {self.tx_n_turns}",
            "",
            "Receiver Configuration:",
            f"  Number of receivers: {self.rx_num}",
            f"  Position (x, y, z): ({self.rx_x}, {self.rx_y}, {self.rx_z}) m",
            "",
            "Separation Configuration:",
            f"  Minimum separation (z-axis): {self.separation_z} m",
            "",
            "Target Configuration:",
            f"  Center (x, y, z): ({self.target_x}, {self.target_y}, {self.target_z}) m",
            f"  Radius: {self.target_radius*1000:.2f} mm",
            f"  Height: {self.target_height*1000:.2f} mm",
            f"  Wall thickness: {self.target_thickness*1000:.2f} mm",
            f"  Conductivity: {self.target_conductivity:.2e} S/m",
            "",
            "Conductivity Model:",
            f"  Air: {self.air_conductivity:.2e} S/m",
            f"  Grass: {self.grass_conductivity:.2e} S/m",
            f"  Soil: {self.soil_conductivity:.2e} S/m",
            f"  Aluminum: {self.aluminum_conductivity:.2e} S/m",
            "",
            "Grass Layer:",
            f"  X range: [{self.grass_x_min}, {self.grass_x_max}] m",
            f"  Z range: [{self.grass_z_min}, {self.grass_z_max}] m",
            "",
            "Mesh Configuration:",
            f"  Base cell size: {self.mesh_dh} m",
            f"  Domain width: {self.mesh_dom_width} m",
            "",
            "Time Configuration:",
            f"  Time channels: {self.time_channel_start*1e6:.1f} to {self.time_channel_end*1e6:.1f} μs",
            f"  Step size: {self.time_channel_step*1e6:.2f} μs",
            f"  Number of channels: {len(self.get_time_channels())}",
            "=" * 60
        ]
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return f"TDEMConfig(config_path='{self.config_path}')"

if __name__ == "__main__":
    config = PiemsolConfig('config.json')
    print(config.summary())
    print(f"\nTarget center: {config.target_center}")
    print(f"Time channels shape: {config.get_time_channels().shape}")
