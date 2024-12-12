import os
import time
import psutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Paths for data
data_dir = "E:/processed_ecg_data"

# Image dimensions
img_height, img_width = 224, 224
batch_size = 32

# Data generator
data_datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)

train_generator = data_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

validation_generator = data_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# Custom CNN model
def create_custom_cnn():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model

model = create_custom_cnn()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the model
print("Starting model training...")
start_time = time.time()
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)
training_time = time.time() - start_time

# Model summary
model.summary()

# Resource usage
process = psutil.Process(os.getpid())
memory_usage = psutil.Process().memory_info().rss / (1024 ** 2)  # Memory in MB
memory_info = process.memory_info()

# Testing the model
print("Evaluating model on test data...")
start_test_time = time.time()
test_loss, test_accuracy = model.evaluate(validation_generator)
test_time = time.time() - start_test_time

# Predictions
predictions = (model.predict(validation_generator) > 0.5).astype(int)
y_true = validation_generator.classes

# Classification metrics
accuracy = accuracy_score(y_true, predictions)
precision = precision_score(y_true, predictions)
recall = recall_score(y_true, predictions)
f1 = f1_score(y_true, predictions)

# Results
print(f"Training Time: {training_time:.2f} seconds")
print(f"Test Time: {test_time:.2f} seconds")
print("Model Summary:")
model.summary()

print("Memory Usage:")
print(f"RSS (Resident Set Size): {memory_info.rss / (1024 * 1024):.2f} MB")
print(f"VMS (Virtual Memory Size): {memory_info.vms / (1024 * 1024):.2f} MB")
print(f"Memory Usage: {memory_usage:.2f} MB")


print("Classification Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
