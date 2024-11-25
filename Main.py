#Main.py
from ECG_Classification import process_dataset
from Custom_CNN_Model import CustomCNNModel
from sklearn.model_selection import train_test_split

# Set up paths and records
data_path = r"E:\İndirilenler\mit-bih-atrial-fibrillation-database-1.0.0\files"  # Replace with actual dataset path
valid_records = ['100', '101', '102']  # Replace with your actual valid record names

# Process dataset and split it into training and testing sets
X, y = process_dataset(data_path, valid_records)

# Split data into train and test sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = CustomCNNModel(input_shape=X_train.shape[1:])
history = model.train(X_train, y_train, X_val, y_val, epochs=10, batch_size=32)

# Evaluate the model
X_test, y_test = X_val, y_val  # Optionally, use a separate test set
evaluation = model.evaluate(X_test, y_test)
print(f'Evaluation results: {evaluation}')
