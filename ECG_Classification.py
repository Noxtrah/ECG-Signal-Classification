#ECG_Classification.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import wfdb
import os

# Function to compute the spectrogram (with or without log transform)
def compute_spectrogram(signal, fs, log_transform=False):
    """
    Computes the spectrogram of a signal.

    Parameters:
        signal (np.array): The input signal (e.g., ECG).
        fs (float): Sampling frequency of the signal.
        log_transform (bool): Whether to apply a log transform to the spectrogram.

    Returns:
        f (np.array): Array of frequency values.
        t (np.array): Array of time values.
        Sxx (np.array): Spectrogram matrix.
    """
    f, t, Sxx = spectrogram(signal, fs=fs, nperseg=256, noverlap=128)
    if log_transform:
        Sxx = 10 * np.log10(Sxx + 1e-10)  # Add a small value to avoid log(0)
    return f, t, Sxx

# Function to get all available records
def get_record_names(data_path):
    records = []
    for file_name in os.listdir(data_path):
        if file_name.endswith('.atr'):
            record_name = file_name.split('.')[0]
            records.append(record_name)
    return records

# Load and process all records
data_path = r"E:\İndirilenler\mit-bih-atrial-fibrillation-database-1.0.0\files"  # Replace with your actual path
records = get_record_names(data_path)

records = records[:3]

# List to store spectrograms
spectrograms_no_log = []
spectrograms_with_log = []
time_axes = []

# Loop through each record and compute its spectrogram
for record_name in records:
    print(f"Processing {record_name}")
    
    try:
        # Load the record and extract the ECG signal
        record = wfdb.rdrecord(f"{data_path}/{record_name}", channels=[0])  # Only the first channel
        ecg_signal = record.p_signal[:, 0]
        fs = record.fs  # Sampling frequency
        
        # Compute spectrograms
        f, t, Sx = compute_spectrogram(ecg_signal, fs, log_transform=False)
        _, _, Sx_log = compute_spectrogram(ecg_signal, fs, log_transform=True)
        
        # Store the spectrograms and time axis for this record
        spectrograms_no_log.append(Sx)
        spectrograms_with_log.append(Sx_log)
        time_axes.append(t)  # Store the time axis for future use
    
    except ValueError as e:
        print(f"Error processing record {record_name}: {e}")

# Ensure the number of spectrograms is the same for both lists
num_records = len(spectrograms_no_log)
if num_records != len(spectrograms_with_log):
    print("Mismatch in number of spectrograms: ", len(spectrograms_no_log), len(spectrograms_with_log))

# Plotting spectrograms for all records
im_list = [spectrograms_no_log, spectrograms_with_log]
im_title = ['Spectrogram without log transform', 'Spectrogram with log transform']

# Increase the figure size for better visibility
fig, ax_list = plt.subplots(2, num_records, figsize=(20, 10))  # Adjusted size

# Loop through all records and plot the spectrograms
for i in range(num_records):
    # Assuming `ax_list` is a 1D list, use only one index to loop over it:
    for j, ax in enumerate(ax_list):  # Remove the extra [:, i]
    # Your plotting logic here

        # Select the spectrogram to plot
        Sxx = im_list[j][i]
        t = time_axes[i]  # Get the time axis for the current record
        
        # Plot the spectrogram
        cax = ax.imshow(Sxx, aspect='auto', cmap='jet', origin='lower')
        ax.set_title(f"{im_title[j]} - {records[i]}", fontsize=14)
        ax.set_xlabel('Time [s]', fontsize=12)
        ax.set_ylabel('Frequency [Hz]', fontsize=12)
        
        # Set x-ticks and y-ticks with appropriate labels
        ax.set_xticks(np.linspace(0, len(t)-1, num=5))
        ax.set_xticklabels([f"{tick:.1f}" for tick in np.linspace(0, max(t), num=5)], fontsize=10)
        ax.set_yticks(np.linspace(0, len(f)-1, num=5))
        ax.set_yticklabels([f"{freq:.1f}" for freq in np.linspace(0, max(f), num=5)], fontsize=10)

        # Add color bar to each plot for better interpretation of the spectrogram values
        fig.colorbar(cax, ax=ax, orientation='vertical')

# Tight layout to avoid overlapping
plt.tight_layout()

# Show the plot
plt.show()



    # def generate_scalogram(self, record_name, wavelet='cmor', scales=np.arange(1, 128)):
    #     record_path = os.path.join(self.data_path, record_name)
    #     record = wfdb.rdrecord(record_path, sampto=5000)
    #     ecg_signal = record.p_signal[:, 0]
    #     sampling_period = 1 / record.fs
    #     coefficients, frequencies = pywt.cwt(ecg_signal, scales, wavelet, sampling_period=sampling_period)
    #     return coefficients
