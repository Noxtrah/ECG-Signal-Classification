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
spectrograms, labels = preprocess_data(data_path)
spectrograms = np.expand_dims(spectrograms, axis=-1)  # Add a channel dimension for CNN

# Convert labels to categorical
num_classes = 2  # Binary classification (AF vs Non-AF)
labels = to_categorical(labels, num_classes=num_classes)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(spectrograms, labels, test_size=0.2, random_state=42)

# Build the CNN model
input_shape = X_train.shape[1:]  # Shape like (128, 128, 1)
num_classes = 2  # Binary classification for AF vs Non-AF

# Initialize and compile the model
model = custom_cnn_model(num_classes=num_classes, input_shape=input_shape)

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
