from tensorflow.keras import layers, models
from tensorflow.keras.models import Model

class custom_cnn_model(Model):
    def __init__(self, num_classes, input_shape):
        super(custom_cnn_model, self).__init__()

       # Using the Input layer to handle input shape
        self.input_layer = layers.Input(shape=input_shape)

        # Convolutional layers with BatchNormalization and Dropout
        # Use 'same' padding to ensure that the output size is not negative after convolution
        self.conv1 = layers.Conv2D(32, (3, 3), padding='same', activation=None)
        self.batch_norm1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()
        self.pool1 = layers.MaxPooling2D((2, 2))
        self.dropout1 = layers.Dropout(0.2)

        self.conv2 = layers.Conv2D(64, (3, 3), padding='same', activation=None)
        self.batch_norm2 = layers.BatchNormalization()
        self.relu2 = layers.ReLU()
        self.pool2 = layers.MaxPooling2D((2, 2))
        self.dropout2 = layers.Dropout(0.2)

        self.conv3 = layers.Conv2D(128, (3, 3), padding='same', activation=None)
        self.batch_norm3 = layers.BatchNormalization()
        self.relu3 = layers.ReLU()
        self.pool3 = layers.MaxPooling2D((2, 2))
        self.dropout3 = layers.Dropout(0.2)

        # Fully connected layers with dropout
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(128, activation='relu')
        self.dropout4 = layers.Dropout(0.5)
        self.dense2 = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.batch_norm1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout4(x)
        return self.dense2(x)

# Specify the correct input shape based on your data
input_shape = (129, 4, 1)  # Update this to match your actual data shape
num_classes = 3  # Replace with the correct number of classes

model = custom_cnn_model(num_classes=num_classes, input_shape=input_shape)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Example summary
model.summary()
