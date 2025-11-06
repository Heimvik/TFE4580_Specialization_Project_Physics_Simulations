import numpy as np


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

    def normalize(self, data, max=None):
        data_min = np.min(data)
        data_max = np.max(data)
        data_normalized = (data - data_min) / (data_max - data_min + 1e-10)
        return data_normalized

    def quantize(self, data, depth, dtype):
        return (np.round(data * 2**depth)).astype(dtype)
    
    def condition_dataset(self, time, decay_curves, labels, label_strings, metadata):
        # Start by scaling it with the area of the loop and the number of windings
        '''
        decay_curves *= (np.pi * self.cfg.tx_radius**2 * self.cfg.tx_n_turns)

        #Clamp it to 0.7V
        decay_curves = self.amplify(time, decay_curves, [[100e-6,10],[300e-6,65]])
        decay_curves = np.clip(decay_curves, 0, 3.3)
        decay_curves = self.quantize(decay_curves, depth=12, dtype=np.uint16)        
        '''
        decay_curves *= (np.pi * self.cfg.tx_radius**2 * self.cfg.tx_n_turns)
        decay_curves = np.clip(decay_curves, 0, 30)
        decay_curves = (self.quantize(decay_curves, depth=14, dtype=np.uint16)/((2**14)-1))*30
        return time, decay_curves, labels, label_strings, metadata