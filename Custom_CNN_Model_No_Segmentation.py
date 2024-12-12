import os
import numpy as np
import wfdb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import time
import psutil  # For memory usage measurement

# Path to dataset
DATA_PATH = 'E:/İndirilenler/mit-bih-atrial-fibrillation-database-1.0.0/files'

# Rhythm annotation mapping
RHYTHM_MAP = {
    "(AFIB": 1,  # Atrial fibrillation (positive class)
    "(AFL": 0,   # Atrial flutter (negative class)
    "(J": 0,     # AV junctional rhythm (negative class)
    "(N": 0      # All other rhythms (negative class)
}

# Define spectrogram size
MAX_TIME = 128
MAX_FREQ = 128


# Step 1: Read and label data
def load_labeled_data(data_path):
    """
    Reads ECG data and rhythm annotations, and assigns binary labels.
    """
    X, y = [], []

    for file in os.listdir(data_path):
        if file.endswith('.dat'):
            record_name = file[:-4]
            record_path = os.path.join(data_path, record_name)

            try:
                # Read the signal and annotations
                record = wfdb.rdrecord(record_path)
                annotations = wfdb.rdann(record_path, 'atr')
                signal = record.p_signal[:, 0]  # Use first channel

                # Label segments using rhythm annotations
                for sample, rhythm in zip(annotations.sample, annotations.aux_note):
                    if rhythm in RHYTHM_MAP:
                        label = RHYTHM_MAP[rhythm]
                        start = max(0, sample - 5000)  # Extract 20-second window
                        end = min(len(signal), sample + 5000)
                        segment = signal[start:end]

                        # Generate spectrogram
                        f, t, Sxx = spectrogram(segment, fs=record.fs, nperseg=1024)
                        Sxx = np.log(Sxx + 1e-10)

                        # Pad or crop spectrogram
                        Sxx = np.pad(Sxx, [(0, max(0, MAX_FREQ - Sxx.shape[0])),
                                           (0, max(0, MAX_TIME - Sxx.shape[1]))],
                                     mode='constant')[:MAX_FREQ, :MAX_TIME]
                        X.append(Sxx)
                        y.append(label)

            except Exception as e:
                print(f"Error processing record {record_name}: {e}")

    return np.array(X), np.array(y)


# Step 2: Build the CNN model
def build_cnn_model(input_shape):
    """
    Builds a simple CNN for binary classification.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Step 3: Train and evaluate the model
def train_and_evaluate():
    """
    Loads the data, trains the CNN model, and evaluates it.
    """
    X, y = load_labeled_data(DATA_PATH)
    print(f"Data shape: {X.shape}, Labels distribution: {np.unique(y, return_counts=True)}")

    if len(np.unique(y)) < 2:
        print("Only one class found in the dataset. Training skipped.")
        return

    # Add channel dimension
    X = X[..., np.newaxis]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Build model
    input_shape = X_train[0].shape
    model = build_cnn_model(input_shape)

    # Record training time
    start_time = time.time()
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=8)
    training_time = time.time() - start_time

    # Record testing time
    start_time = time.time()
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    test_time = time.time() - start_time

    # Evaluate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=1)
    recall = recall_score(y_test, y_pred, zero_division=1)
    f1 = f1_score(y_test, y_pred, zero_division=1)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Test Time: {test_time:.2f} seconds")

    # Resource metrics
    total_params = model.count_params()
    trainable_params = np.sum([np.prod(v.shape) for v in model.trainable_variables])
    non_trainable_params = total_params - trainable_params
    memory_usage = psutil.Process().memory_info().rss / (1024 ** 2)  # Memory in MB

    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")
    print(f"Non-Trainable Parameters: {non_trainable_params}")
    print(f"Memory Usage: {memory_usage:.2f} MB")

    # Plot accuracy
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Test Accuracy')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    train_and_evaluate()
