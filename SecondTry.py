import wfdb
import numpy as np

# Load the ECG signal from the MIT-BIH AF Database
record = wfdb.rdrecord('E:/İndirilenler/mit-bih-atrial-fibrillation-database-1.0.0/files/04015', channels=[0])  # Specify the channel
ecg_signal = record.p_signal.flatten()  # Flatten the signal for easy handling


import scipy.signal as signal
import matplotlib.pyplot as plt

# Generate a spectrogram of the ECG signal
frequencies, times, Sxx = signal.spectrogram(ecg_signal, fs=record.fs, nperseg=1024)

# Plot and save the spectrogram
plt.pcolormesh(times, frequencies, np.log(Sxx))
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('ECG Spectrogram')
plt.colorbar(label='Log Power Spectral Density')
plt.savefig('ecg_spectrogram.png')
plt.show()


import wfdb
import numpy as np
import scipy.signal as signal
import os

# Define a fixed size for the spectrogram
MAX_TIME = 128  # Maximum number of time steps
MAX_FREQ = 128  # Maximum number of frequency bins

import wfdb
import numpy as np
import scipy.signal as signal
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt

# Define a fixed size for the spectrogram
MAX_TIME = 128  # Maximum number of time steps
MAX_FREQ = 128  # Maximum number of frequency bins

# Define function to load and preprocess dataset
def load_dataset(data_path):
    X = []  # Spectrograms (input data)
    y = []  # Labels (AF or Non-AF)

    # Iterate through the dataset files
    for file_name in os.listdir(data_path):
        if file_name.endswith('.dat'):
            record = wfdb.rdrecord(os.path.join(data_path, file_name[:-4]), channels=[0])
            ecg_signal = record.p_signal.flatten()
            frequencies, times, Sxx = signal.spectrogram(ecg_signal, fs=record.fs, nperseg=1024)
            
            # Take log of the spectrogram to avoid negative values
            log_Sxx = np.log(Sxx + 1e-10)  # Adding a small value to avoid log(0)

            # Pad or truncate the spectrogram to the fixed size
            if log_Sxx.shape[1] > MAX_TIME:
                log_Sxx = log_Sxx[:, :MAX_TIME]  # Crop time axis if it's too long
            elif log_Sxx.shape[1] < MAX_TIME:
                pad_width = MAX_TIME - log_Sxx.shape[1]
                log_Sxx = np.pad(log_Sxx, ((0, 0), (0, pad_width)), mode='constant')  # Pad time axis

            if log_Sxx.shape[0] > MAX_FREQ:
                log_Sxx = log_Sxx[:MAX_FREQ, :]  # Crop frequency axis if it's too long
            elif log_Sxx.shape[0] < MAX_FREQ:
                pad_width = MAX_FREQ - log_Sxx.shape[0]
                log_Sxx = np.pad(log_Sxx, ((0, pad_width), (0, 0)), mode='constant')  # Pad frequency axis

            X.append(log_Sxx)
            
            # Assuming the label is inferred from the filename, e.g., AF or Non-AF
            y.append(0 if 'Non-AF' in file_name else 1)

    return np.array(X), np.array(y)

# Load dataset
X, y = load_dataset('E:/İndirilenler/mit-bih-atrial-fibrillation-database-1.0.0/files')

# Check the shape of X to ensure all spectrograms have the same dimensions
print(X.shape)

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape the data for CNN input
input_shape = X_train[0].shape + (1,)  # Add a channel dimension for grayscale images

# Check reshaped data
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# Build a custom CNN model
def build_cnn(input_shape):
    model = tf.keras.Sequential()

    # Add convolutional layers
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    # Flatten the output from the convolutional layers
    model.add(tf.keras.layers.Flatten())

    # Add fully connected layers
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))  # Binary classification (AF or Non-AF)

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Build the CNN model
model = build_cnn(input_shape)
model.summary()

# Reshape the data for the CNN
X_train = X_train.reshape((-1, *input_shape))  # Add channel dimension for each input
X_test = X_test.reshape((-1, *input_shape))    # Add channel dimension for each input

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_acc * 100:.2f}%')

# Plot accuracy and loss
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
