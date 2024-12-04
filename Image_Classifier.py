import wfdb
import os

# Path to the folder containing the records
folder_path = r"E:/İndirilenler/mit-bih-atrial-fibrillation-database-1.0.0/files"

# Function to classify a record as AF or Non-AF
def classify_record(record_path):
    # Load annotations
    annotations = wfdb.rdann(record_path, 'atr')
    
    # Check if 'AFIB' is in the annotations
    if any('AFIB' in annotation for annotation in annotations.aux_note):
        return "AF"
    else:
        return "Non-AF"

# Iterate over all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".atr"):
        # Get the base name of the record (without file extension)
        record_base_name = os.path.splitext(filename)[0]
        
        # Build the full path to the record
        record_path = os.path.join(folder_path, record_base_name)
        
        # Classify the record
        classification = classify_record(record_path)
        
        # Output the classification
        print(f"Record {record_base_name} is classified as: {classification}")
