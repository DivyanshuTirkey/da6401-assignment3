#!/bin/bash

# Download the Dakshina dataset
echo "Downloading Dakshina dataset..."
wget https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar

# Extract the dataset
echo "Extracting dataset..."
tar -xf dakshina_dataset_v1.0.tar

# Create data directory if it doesn't exist
mkdir -p data

# Move Hindi transliteration data to data directory
echo "Organizing data files..."
cp -r dakshina_dataset_v1.0/hi/lexicons/* data/

echo "Data download complete!"
