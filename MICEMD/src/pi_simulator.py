import numpy as np
import MicEMD.tdem as tdem
from MicEMD.preprocessor import data_prepare
from MicEMD.handler import TDEMHandler
import math
import matplotlib.pyplot as plt

# the attribute of the steel, Ni, and Al including permeability, permeability of vacuum and conductivity
attribute = np.array([[1.000022202, 4*math.pi*1e-7, 37667620.91],[696.3028547, 4*math.pi*1e-7, 50000000], [99.47183638, 4*math.pi*1e-7, 14619883.04]])
# create and initial the target, detector, collection class of Tdem
target = tdem.Target(material=['Al', 'Ni', 'Steel'], shape=['Oblate spheroid', 'Prolate spheroid'],
                     attribute=attribute, ta_min=0.01, ta_max=1.5, tb_min=0.01, tb_max=1.5, a_r_step=0.08,
                     b_r_step=0.08)
detector = tdem.Detector(0.4, 20, 0, 0)
collection = tdem.Collection(t_split=20, snr=30)

# call the simulate interface, the forward_result is a tuple which conclude the Sample Set and a random
# sample of Sample Set, the random sample of Sample Set is used to visualize
fwd_res = tdem.simulate(target, detector, collection, model='dipole')


class Preprocessor:
    @staticmethod
    def normalize(data):
        return data / np.max(np.abs(data))
    
    @staticmethod
    def quantize(data, resolution):
        data = Preprocessor.normalize(data)
        return np.round(data * (2**resolution - 1))
    
    @staticmethod
    def amplify(data, time, gain_at_time):
        amplified_data = data.copy()
        for i in range(len(data)):
            for j in range(len(gain_at_time['Time'])):
                if time[i] >= gain_at_time['Time'][j]:
                    amplified_data[i] *= gain_at_time['Gain'][j]
        return amplified_data

# Plot the forward results on a linear time scale
def plot_fwd_res_linear_time(fwd_res):
    """
    Plot the forward simulation results on a linear time scale
    
    Parameters:
    fwd_res: tuple containing (feature_lable, sample)
        - feature_lable: array with simulation results for all samples
        - sample: dict with sample data including time 't', 'M1', 'M2' responses
    """
    sample_data = fwd_res[1]  # Get the sample data
        
    if sample_data is not None:
        # Extract time and response data
        preprocessor = Preprocessor()
        M1 = sample_data['M1']
        amplified_M1 = preprocessor.amplify(M1, sample_data['t'], gain_at_time={'Time': [0.02], 'Gain': [10]})
        quantized_M1 = preprocessor.quantize(M1, resolution=8)  # Example: 8-bit quantization
        print("M1 (original)    Quantized M1")
        for orig, quant in zip(M1, quantized_M1):
            print(f"{int(orig):d}    {int(quant):d}")
        t = sample_data['t']  # Time array
        M1 = amplified_M1  # Response M1 (with noise)
        M2 = sample_data['M2']  # Response M2 (with noise)
        M1_clean = sample_data['M1_without_noise']  # Clean response M1
        M2_clean = sample_data['M2_without_noise']  # Clean response M2
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot M1 responses on linear time scale
        ax1.plot(t, M1_clean, 'b-', linewidth=2, label='M1 (without noise)', alpha=0.8)
        ax1.plot(t, M1, 'b--', linewidth=1, label='M1 (with noise)', alpha=0.6)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Response M1')
        ax1.set_title(f'TDEM Forward Response M1 - Linear Time Scale\n'
                      f'Material: {sample_data["material"]}, ta: {sample_data["ta"]:.3f}m, '
                      f'tb: {sample_data["tb"]:.3f}m, SNR: {sample_data["SNR"]}dB')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_xlim(left=0)  # Start from 0 for linear scale
        
        # Plot M2 responses on linear time scale
        ax2.plot(t, M2_clean, 'r-', linewidth=2, label='M2 (without noise)', alpha=0.8)
        ax2.plot(t, M2, 'r--', linewidth=1, label='M2 (with noise)', alpha=0.6)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Response M2')
        ax2.set_title('TDEM Forward Response M2 - Linear Time Scale')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_xlim(left=0)  # Start from 0 for linear scale
        
        plt.tight_layout()
        plt.show()
        
        # Also create a combined plot
        fig2, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.plot(t, M1_clean, 'b-', linewidth=2, label='M1 (without noise)', alpha=0.8)
        ax.plot(t, M1, 'b--', linewidth=1, label='M1 (with noise)', alpha=0.6)
        ax.plot(t, M2_clean, 'r-', linewidth=2, label='M2 (without noise)', alpha=0.8)
        ax.plot(t, M2, 'r--', linewidth=1, label='M2 (with noise)', alpha=0.6)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Response')
        ax.set_title(f'TDEM Forward Responses - Linear Time Scale\n'
                     f'Material: {sample_data["material"]}, ta: {sample_data["ta"]:.3f}m, '
                     f'tb: {sample_data["tb"]:.3f}m, SNR: {sample_data["SNR"]}dB')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim(left=0)  # Start from 0 for linear scale
        
        plt.tight_layout()
        plt.show()
        
        print(f"Plotted TDEM responses for {sample_data['material']} target")
        print(f"Time range: {t.min():.6f} to {t.max():.3f} seconds")
        print(f"M1 response range: {M1_clean.min():.6e} to {M1_clean.max():.6e}")
        print(f"M2 response range: {M2_clean.min():.6e} to {M2_clean.max():.6e}")
    else:
        print("No sample data available for plotting")

# Plot the forward results
plot_fwd_res_linear_time(fwd_res)
'''
# split data sets and normalization for the Sample Set, Here we classify materials
ori_dataset_material = data_prepare(fwd_res[0], task='material')

# dimensionality reduction, return a tuple conclude train_set and test_set
dim_dataset_material = tdem.preprocess(ori_dataset_material, dim_red_method='PCA', n_components=20)

# parameters setting of the classification model by dict
para = {'solver': 'lbfgs', 'hidden_layer_sizes': (50,), 'activation': 'tanh'}

# call the classify interface
# the res of the classification which is a dict that conclude accuracy, predicted value and true value
cls_material_res = tdem.classify(dim_dataset_material, 'ANN', para)


# create the TDEMHandler and call the methods to show and save the results
# set the TDEMHandler without parameters to save the results
# the file path of the results is generated by your settings
handler = TDEMHandler()

# save the forward results and one sample data
handler.save_fwd_data(fwd_res[0], file_name='fwd_res.csv')
handler.save_sample_data(fwd_res[1], file_name='sample.csv', show=True)

# save the original dataset that distinguishes material
handler.save_fwd_data(ori_dataset_material[0], file_name='ori_material_train.csv')
handler.save_fwd_data(ori_dataset_material[1], file_name='ori_material_test.csv')

# save the final dataset after dimensionality reduction
handler.save_fwd_data(dim_dataset_material[0], file_name='dim_material_train.csv')
handler.save_fwd_data(dim_dataset_material[1], file_name='dim_material_test.csv')

# save the classification results
handler.show_cls_res(cls_material_res, ['Steel', 'Ni', 'Al'], show=True, save=True, file_name='cls_result_material.pdf')
handler.save_cls_res(cls_material_res, 'cls_material_res.csv')


# classify the shape of the targets
ori_dataset_shape = data_prepare(fwd_res[0], task='shape')
dim_dataset_shape = tdem.preprocess(ori_dataset_shape, dim_red_method='PCA', n_components=20)
cls_shape_res = tdem.classify(dim_dataset_shape, 'ANN', para)
# save the original dataset that distinguishes material
handler.save_fwd_data(ori_dataset_shape[0], file_name='ori_shape_train.csv')
handler.save_fwd_data(ori_dataset_shape[1], file_name='ori_shape_test.csv')
# save the final dataset after dimensionality reduction
handler.save_fwd_data(dim_dataset_shape[0], file_name='dim_shape_train.csv')
handler.save_fwd_data(dim_dataset_shape[1], file_name='dim_shape_test.csv')
handler.show_cls_res(cls_shape_res, ['Oblate spheroid', 'Prolate spheroid'], show=True, save=True, file_name='cls_result_shape.pdf')
handler.save_cls_res(cls_shape_res, 'cls_shape_res.csv')
'''