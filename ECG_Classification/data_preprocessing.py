import os
import numpy as np
import wfdb
from sklearn.preprocessing import MinMaxScaler
from scipy import signal
import pywt

def get_record_names(directory):
    """
    Returns a list of the base names (without extension) of the ECG records in the specified directory.
    """
    record_names = []
    file_list = os.listdir(directory)
    for file_name in file_list:
        base_name, ext = os.path.splitext(file_name)
        if ext == '.dat':  # or any other condition to check valid records
            record_names.append(base_name)
    return record_names


def apply_spectrogram(signal_data, fs=360):
    """
    Converts the ECG signal to a spectrogram.
    """
    f, t, Sxx = signal.spectrogram(signal_data, fs)
    return Sxx


def apply_wavelet_transform(signal_data):
    """
    Applies a wavelet transform (using the Discrete Wavelet Transform) to the ECG signal.
    """
    coeffs = pywt.wavedec(signal_data, 'db4', level=5)  # 'db4' is a common wavelet
    # Use the approximation coefficients (the first level coefficients) for simplicity
    return coeffs[0]  # You can change this to use other levels as needed


def preprocess_data(directory, use_spectrogram=True):
    """
    Preprocesses the ECG data from the given directory and returns the features and labels.
    
    Args:
        directory (str): Path to the directory containing ECG data files.
        use_spectrogram (bool): If True, uses spectrograms. If False, uses wavelet transforms.
    
    Returns:
        features (np.ndarray): Preprocessed features from the ECG data.
        labels (np.ndarray): Labels associated with the ECG data.
    """
    features = []
    labels = []
    
    # List all files in the directory
    file_list = os.listdir(directory)
    
    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    # Process each file in the directory
    for file_name in file_list:
        # Ensure file extensions are handled correctly
        base_file_name, ext = os.path.splitext(file_name)

        if ext == ".atr" or ext == ".qrs":
            # Check if the .hea file exists
            hea_file = os.path.join(directory, f"{base_file_name}.hea")
            
            if not os.path.exists(hea_file):
                print(f"Missing .hea file for {base_file_name}. Skipping this file.")
                continue

            # Read the signal data using WFDB
            try:
                record = wfdb.rdrecord(os.path.join(directory, base_file_name), sampfrom=0)
            except ValueError as e:
                print(f"Error reading record for {base_file_name}: {e}")
                continue

            # Read the annotation data (if exists)
            try:
                annotation = wfdb.rdann(os.path.join(directory, base_file_name), 'atr')
            except Exception as e:
                print(f"Error reading annotation for {base_file_name}: {e}")
                continue

            # Normalize the signal data
            normalized_signal = scaler.fit_transform(record.p_signal)

            # Pad or truncate the signal to a fixed length
            MAX_LENGTH = 1000  # Set a fixed length you want for all signals
            if len(normalized_signal.flatten()) < MAX_LENGTH:
                padded_signal = np.pad(normalized_signal.flatten(), (0, MAX_LENGTH - len(normalized_signal.flatten())), 'constant')
            else:
                # If the signal is longer than MAX_LENGTH, truncate it
                padded_signal = normalized_signal.flatten()[:MAX_LENGTH]

            # Apply transformation (spectrogram or wavelet)
            if use_spectrogram:
                processed_signal = apply_spectrogram(padded_signal)
            else:
                processed_signal = apply_wavelet_transform(padded_signal)

            # Reshape the processed signal for CNN input (ensure it's 2D or 3D)
            features.append(processed_signal)

            # Extract label (for example, we can assume the annotation contains a label for AF or non-AF)
            label = 1 if 'AF' in annotation.symbol else 0
            labels.append(label)
    
    # Convert features and labels to numpy arrays for machine learning compatibility
    features = np.array(features)
    labels = np.array(labels)

    return features, labels


# Example usage:
directory = r"E:\İndirilenler\mit-bih-atrial-fibrillation-database-1.0.0\files"

from data_preprocessing import preprocess_data, get_record_names
from cnn_model import custom_cnn_model
from utils import plot_training_history
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np

# Define paths and parameters
data_path = r"E:\İndirilenler\mit-bih-atrial-fibrillation-database-1.0.0\files"  # Dataset path
records = get_record_names(data_path)

# Preprocess the data
spectrograms, labels = preprocess_data(data_path, use_spectrogram=True)  # Change to False to use wavelets
spectrograms = np.expand_dims(spectrograms, axis=-1)  # Add a channel dimension for CNN

# Convert labels to categorical
num_classes = 2  # Binary classification (AF vs Non-AF)
labels = to_categorical(labels, num_classes=num_classes)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(spectrograms, labels, test_size=0.2, random_state=42)

# Build the CNN model
input_shape = X_train.shape[1:]  # Shape of a single spectrogram or wavelet transform
model = custom_cnn_model(input_shape, num_classes)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=8
)

# Plot training history
plot_training_history(history)

