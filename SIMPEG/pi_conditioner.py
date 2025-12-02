import numpy as np
import os
import matplotlib.pyplot as plt


class PiConditioner:
    def __init__(self, cfg):
        self.cfg = cfg
        
    def add_noise(self, time, data, late_time, snr_db):
        if isinstance(time, list):
            time = time[0]
        
        if hasattr(time, 'ndim') and time.ndim > 1:
            time = time[0]
        
        print(f"Time range: {time[0]:.6e} to {time[-1]:.6e}")
        
        if late_time < time[0]:
            raise ValueError("late_time must be within the range of time array")
        if late_time > time[-1]:
            raise ValueError("late_time must be within the range of time array")

        idx = np.where(time >= late_time)[0]
        if len(idx) == 0:
            raise ValueError("late_time is beyond the maximum time in the array")
        idx = idx[0]

        signal_power = np.mean(data[idx:]**2)
        snr_linear = 10**(snr_db / 10)
        noise_power = signal_power / snr_linear
        noise_std = np.sqrt(noise_power)

        noise = np.random.normal(0, noise_std, size=data.shape)
        return data + noise
        

    def amplify(self, time, data, time_gain):
        if isinstance(time, list):
            time = time[0]
        
        if hasattr(time, 'ndim') and time.ndim > 1:
            time = time[0]
        
        time_gain = np.array(time_gain)
        if time_gain.size > 0:
            time_gain = time_gain[np.argsort(time_gain[:, 0])]
        gain = np.ones_like(time)
        for t, g in time_gain:
            mask = time >= t
            gain[mask] = g
        return data * gain

    def normalize(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-10)

    def quantize(self, data, depth, dtype):
        data = self.normalize(data)
        return (np.round(data * 2**depth)).astype(dtype)
    
    def condition_dataset(self, I_0, time):
        time = time[-1024:]
        if I_0.ndim > 1:
            I_0 = I_0[:, -1024:]
        else:
            I_0 = I_0[-1024:]
        v_0 = np.clip(I_0,  1e-10, 0.7)
        v_1 = 0.2*np.log10(v_0/1e-6)
        v_2 = np.clip(v_1, 0, 3.3)
        v_3 = self.quantize(v_2, depth=8, dtype=np.uint16)       
        '''
        # Plot conditioning stages for the first sample in a 3x2 matrix
        if I_0.ndim > 1:
            sample_idx = 0
            i0_plot = I_0[sample_idx]
            v0_plot = v_0[sample_idx]
            v1_plot = v_1[sample_idx]
            v2_plot = v_2[sample_idx]
            v3_plot = v_3[sample_idx]
        else:
            i0_plot = I_0
            v0_plot = v_0
            v1_plot = v_1
            v2_plot = v_2
            v3_plot = v_3

        fig, axes = plt.subplots(3, 2, figsize=(32, 30), sharex=True)
        axes = axes.flatten()  # Flatten to 1D array for easier indexing
        
        os.makedirs('Images/Plots', exist_ok=True)
        
        # Plot I_0(t) - Input
        fig1, ax1 = plt.subplots(figsize=(16, 10))
        ax1.plot(time * 1e6, i0_plot, 'k-', linewidth=2)
        ax1.set_title(r'$I_0(t)$', fontsize=50)
        ax1.set_ylabel(r'Voltage (V)', fontsize=45)
        ax1.set_xlabel(r'Time ($\mu$s)', fontsize=45)
        ax1.tick_params(axis='both', which='major', labelsize=45)
        ax1.grid(True, alpha=0.3)
        fig1.savefig('Images/Plots/I_0_input.png', bbox_inches='tight', pad_inches=0.5)
        plt.close(fig1)
        
        # Plot V_0(t)
        fig2, ax2 = plt.subplots(figsize=(16, 10))
        ax2.plot(time * 1e6, v0_plot, 'b-', linewidth=2)
        ax2.set_title(r'$v_0(t)$', fontsize=50)
        ax2.set_ylabel(r'Voltage (V)', fontsize=45)
        ax2.set_xlabel(r'Time ($\mu$s)', fontsize=45)
        ax2.tick_params(axis='both', which='major', labelsize=45)
        ax2.grid(True, alpha=0.3)
        fig2.savefig('Images/Plots/V_0.png', bbox_inches='tight', pad_inches=0.5)
        plt.close(fig2)
        
        # Plot V_1(t)
        fig3, ax3 = plt.subplots(figsize=(16, 10))
        ax3.plot(time * 1e6, v1_plot, 'r-', linewidth=2)
        ax3.set_title(r'$v_1(t)$', fontsize=50)
        ax3.set_ylabel(r'Voltage (V)', fontsize=45)
        ax3.set_xlabel(r'Time ($\mu$s)', fontsize=45)
        ax3.tick_params(axis='both', which='major', labelsize=45)
        ax3.grid(True, alpha=0.3)
        fig3.savefig('Images/Plots/V_1.png', bbox_inches='tight', pad_inches=0.5)
        plt.close(fig3)
        
        # Plot V_2(t)
        fig4, ax4 = plt.subplots(figsize=(16, 10))
        ax4.plot(time * 1e6, v2_plot, 'g-', linewidth=2)
        ax4.set_title(r'$v_2(t)$', fontsize=50)
        ax4.set_ylabel(r'Voltage (V)', fontsize=45)
        ax4.set_xlabel(r'Time ($\mu$s)', fontsize=45)
        ax4.tick_params(axis='both', which='major', labelsize=45)
        ax4.grid(True, alpha=0.3)
        fig4.savefig('Images/Plots/V_2.png', bbox_inches='tight', pad_inches=0.5)
        plt.close(fig4)
        
        # Plot V_3(t)
        fig5, ax5 = plt.subplots(figsize=(16, 10))
        ax5.plot((time * 1e6)[-1024:], v3_plot[-1024:], 'm-', linewidth=2)
        ax5.set_title(r'$I_1[n]$', fontsize=50)
        ax5.set_xlabel(r'Time ($\mu$s)', fontsize=45)
        ax5.tick_params(axis='both', which='major', labelsize=45)
        ax5.grid(True, alpha=0.3)
        fig5.savefig('Images/Plots/V_3.png', bbox_inches='tight', pad_inches=0.5)
        plt.close(fig5)
        '''

        return v_3, time