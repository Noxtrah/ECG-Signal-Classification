import os
import time
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import psutil

# Define paths
spectrogram_dir = "E:/spectrogram_images/AFIB"

# Set image parameters
img_height = 224
img_width = 224
batch_size = 16

# Load images and preprocess
image_files = [f for f in os.listdir(spectrogram_dir) if f.endswith('.png')]  # Assuming spectrogram images are PNG
images = []
labels = []

# Read and resize images
for image_file in image_files:
    img_path = os.path.join(spectrogram_dir, image_file)
    img = load_img(img_path, target_size=(img_height, img_width))  # Use load_img here
    img_array = img_to_array(img) / 255.0  # Normalize pixel values using img_to_array
    images.append(img_array)
    labels.append(1)  # All images are AFIB (class label 1)

# Convert to numpy arrays
X_data = np.array(images)
y_labels = np.array(labels) 

# Split the data into training and validation sets (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X_data, y_labels, test_size=0.2, random_state=42)

# Measure Training Time
start_train_time = time.time()

# Load VGG16 model pre-trained on ImageNet, excluding the top (fully connected) layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom layers on top
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)  # Since you have only one class (AFIB)

# Define the model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of the pre-trained VGG16 model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Measure Test Time
start_test_time = time.time()

# Evaluate the model on the validation set
y_pred = model.predict(X_val)
y_pred_class = (y_pred > 0.5).astype(int)  # Convert probabilities to binary labels

# Measure Test Time
end_test_time = time.time()

# Calculate the performance metrics
accuracy = accuracy_score(y_val, y_pred_class)
precision = precision_score(y_val, y_pred_class)
recall = recall_score(y_val, y_pred_class)
f1 = f1_score(y_val, y_pred_class)

# Training and test time
end_train_time = time.time()
train_time = end_train_time - start_train_time
test_time = end_test_time - start_test_time

# Memory Usage and Parameter Count
model.summary()  # Prints the model's parameter count

# Measure memory usage using psutil (can be used during model inference)
process = psutil.Process(os.getpid())
memory_usage = process.memory_info().rss  # in bytes

# Output the results
print(f"Training Time: {train_time:.2f} seconds")
print(f"Test Time: {test_time:.2f} seconds")
print(f"Memory Usage: {memory_usage / (1024 * 1024):.2f} MB")  # Convert to MB
print(f"Parameter Count: {model.count_params()} parameters")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")