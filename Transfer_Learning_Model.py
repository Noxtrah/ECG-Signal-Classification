from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout

class TransferLearningModel:
    def __init__(self, input_shape):
        self.base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        self.model = self._build_model()

    def _build_model(self):
        x = Flatten()(self.base_model.output)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        output = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=self.base_model.input, outputs=output)
        return model

    def freeze_base_model(self):
        for layer in self.base_model.layers:
            layer.trainable = False

    def compile_model(self, optimizer='adam'):
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
        history = self.model.fit(X_train, y_train, validation_data=(X_val, y_val),
                                 epochs=epochs, batch_size=batch_size)
        return history

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)
