#!/bin/bash

# Basic settings for the model
models=("resnet" "cnn") # "cnn"
datasets=("ECG5000" "FordA" "Wafer") # "FordB"
new="False"

# Base path for the processed data
path="/data/"

for model in "${models[@]}"; do

    for dataset in "${datasets[@]}"; do

        # Convert model and dataset names to lower case
        model_lowercase=$(echo "$model" | tr '[:upper:]' '[:lower:]')
        dataset_lowercase=$(echo "$dataset" | tr '[:upper:]' '[:lower:]')

        echo "BASH - Looking for $model_lowercase-$dataset_lowercase.pt"

        # Check how large the files are and what is even in the subdirectories
        # du -ah "$path"

        # Train the model if not available
        if ! ls "$path$model_lowercase-$dataset_lowercase.pt" 1> /dev/null 2>&1; then
            echo "BASH - Model or dataset not found. Calling Python script..."
            python train_model.py -d "$dataset" -m "$model" -p "$path"
            echp ""
        else
            echo "BASH - Model and dataset exist."
            echp ""
        fi

        # Extract data from the model
        python extract_data.py -d "$dataset" -m "$model" -p "$path" -cn "$new"

        # Fix permissions to standard user
        if [ ! -z "$HOST_UID" ] && [ ! -z "$HOST_GID" ]; then
            chown -R $HOST_UID:$HOST_GID "$path"
        fi

        echo "BASH - Extraction done."
        echo ""

    done
done
