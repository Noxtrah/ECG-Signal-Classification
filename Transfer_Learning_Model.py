import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras import layers, models
import os

# Simulate dataset creation (replace this with your actual data loading process)
def generate_dummy_data():
    np.random.seed(42)
    X = np.random.rand(100, 224, 224, 3)  # Dummy spectrograms (100 samples of 224x224x3 images)
    y = np.random.randint(0, 2, 100)  # Binary labels (0 or 1)
    return X, y

# Load data
X, y = generate_dummy_data()

# Split into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Define a simple transfer learning model using MobileNetV2
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False  # Freeze the base model

# Add custom layers
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=16
)

# Evaluate the model on the test set
y_pred_probs = model.predict(X_test)  # Probabilities
y_pred_binary = (y_pred_probs > 0.5).astype(int)  # Convert to binary labels

# Calculate metrics
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_binary))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_binary))

# Additional metrics
accuracy = accuracy_score(y_test, y_pred_binary) * 100
precision = precision_score(y_test, y_pred_binary, zero_division=0)
recall = recall_score(y_test, y_pred_binary, zero_division=0)
f1 = f1_score(y_test, y_pred_binary, zero_division=0)

print(f"Accuracy: {accuracy:.2f}%")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Print training and test times (dummy values here, replace with actual timing logic if needed)
import time
train_time = time.time() - history.epoch[0]
test_time = time.time() - train_time

print(f"Training Time: {train_time:.2f} seconds")
print(f"Test Time: {test_time:.2f} seconds")

# Model summary and parameters
total_params = model.count_params()
trainable_params = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
non_trainable_params = np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])

print(f"Total Parameters: {total_params}")
print(f"Trainable Parameters: {trainable_params}")
print(f"Non-Trainable Parameters: {non_trainable_params}")
