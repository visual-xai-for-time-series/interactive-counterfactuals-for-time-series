#!/bin/bash

# Basic settings for the model
model="resnet"
dataset="ECG5000"
path="/data/"

model_lowercase=$(echo "$model" | tr '[:upper:]' '[:lower:]')
dataset_lowercase=$(echo "$dataset" | tr '[:upper:]' '[:lower:]')

du -ah "$path"

# Train the model if not available
if ! ls "$path$model_lowercase-$dataset_lowercase.pt" 1> /dev/null 2>&1; then
    echo "Model or dataset not found. Calling Python script..."
    python train_model.py -d "$dataset" -m "$model" -p "$path"
else
    echo "Model and dataset exist."
fi

# Extract data from the model
python extract_data.py -d "$dataset" -m "$model" -p "$path"

# Fix permissions to standard user
chown -R 1000:1000 "$path"
