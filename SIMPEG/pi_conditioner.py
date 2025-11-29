import numpy as np


class PiConditioner:
    def __init__(self, cfg):
        self.cfg = cfg

    def add_noise(self, data, snr_db):
        noise_std = np.sqrt(np.mean(data**2) / 10**(snr_db / 10))
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
        data_normalized = (data - data.min()) / (data.max() - data.min() + 1e-10)
        data_clean = np.nan_to_num(data_normalized, nan=0.0, posinf=0.0, neginf=0.0)
        return (np.round(data_clean * (2**depth - 1))).astype(dtype)
    
    def condition_dataset(self, I_0):
        
        v_0 = np.clip(I_0, 1e-10, 30)
        v_1 = np.log10(v_0)
        v_2 = np.clip(v_1, 1 + (1 / 2**12) * (3.3 - 1), 3.3)
        I_1 = self.quantize(v_2, depth=12, dtype=np.uint16)        

        return I_1