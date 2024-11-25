import numpy as np
from cnn_model import custom_cnn_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# Assuming you have your training and validation data prepared
# Replace these with your actual data paths or arrays
train_data = np.random.rand(100, 128, 128, 1)  # Example shape (100 samples, 128x128 images, 1 channel)
train_labels = np.random.randint(0, 2, size=(100,))  # Binary labels

val_data = np.random.rand(20, 128, 128, 1)  # Example validation data
val_labels = np.random.randint(0, 2, size=(20,))  # Validation labels

# Initialize the custom CNN model with the input shape
model = custom_cnn_model(input_shape=(128, 128, 1))  # Adjust the input shape as per your spectrogram/wavelet dimensions
model.compile_model()

# Create data generators for training and validation
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow(train_data, train_labels, batch_size=32)

val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow(val_data, val_labels, batch_size=32)

# Early stopping to avoid overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.model.fit(
    train_generator,
    epochs=50,
    validation_data=val_generator,
    callbacks=[early_stopping]
)
