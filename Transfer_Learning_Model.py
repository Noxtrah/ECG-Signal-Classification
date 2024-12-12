# Function to generate and save spectrogram images
# def generate_spectrogram(segment, output_folder, label, idx):
#     # Ensure the output folder exists
#     os.makedirs(output_folder, exist_ok=True)

#     # Calculate the spectrogram using scipy's spectrogram function
#     f, t, Sxx = spectrogram(segment, fs=1000)  # Adjust fs (sampling frequency) if needed
    
#     # Save the spectrogram image directly to the output folder
#     output_path = os.path.join(output_folder, f"segment_{idx}.png")
    
#     # Generate the spectrogram and save it
#     plt.figure(figsize=(10, 4))
#     plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')  # Log scale for intensity
#     plt.colorbar(label='Intensity [dB]')
#     plt.xlabel('Time [s]')
#     plt.ylabel('Frequency [Hz]')
#     plt.title(f'Spectrogram of Segment {idx}')
#     plt.savefig(output_path)
#     plt.close()  # Close the plot to free memory

import os
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
import psutil
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Directory paths
data_dir = "E:/processed_ecg_data"

# Transfer Learning Model creation
def create_transfer_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False
    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Binary classification (AFIB or Non-AF)
    ])
    return model

# Train and evaluate the model
def train_and_evaluate_model(data_dir):
    datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)

    # Train and validation generators
    train_gen = datagen.flow_from_directory(data_dir, target_size=(224, 224), batch_size=32, subset='training', class_mode='binary')
    val_gen = datagen.flow_from_directory(data_dir, target_size=(224, 224), batch_size=32, subset='validation', class_mode='binary')

    # Train the model
    model = create_transfer_model()
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    
    start_time = time.time()
    model.fit(train_gen, validation_data=val_gen, epochs=10)
    training_time = time.time() - start_time
    
    print(f"\nTraining Time: {training_time:.2f} seconds")
    
    # Log resources
    log_resource_usage(model)
    
    # Evaluate on validation set
    evaluate_model(model, val_gen)
    return model

# Function to evaluate the model and print metrics
def evaluate_model(model, test_gen):
    start_time = time.time()
    y_true = test_gen.classes  # Ground truth labels
    y_pred_probs = model.predict(test_gen)  # Predicted probabilities
    y_pred = (y_pred_probs > 0.5).astype("int32").flatten()  # Convert to binary predictions
    
    end_time = time.time()
    test_time = end_time - start_time
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print("\nClassification Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Test Time: {test_time:.2f} seconds")

# Function to log resource usage and parameter count
def log_resource_usage(model):
    # Print model summary for parameter count
    print("\nModel Summary:")
    model.summary()

    # Log memory usage
    process = psutil.Process()
    memory_info = process.memory_info()
    print("\nMemory Usage:")
    print(f"RSS (Resident Set Size): {memory_info.rss / (1024 ** 2):.2f} MB")
    print(f"VMS (Virtual Memory Size): {memory_info.vms / (1024 ** 2):.2f} MB")

# Main script
print("Starting model training...")
trained_model = train_and_evaluate_model(data_dir)
print("Model training complete.")

