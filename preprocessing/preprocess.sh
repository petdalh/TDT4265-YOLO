#!/bin/bash

# Data Preprocessing Script
# This script runs the data cleaning and augmentation scripts sequentially

echo "Starting data preprocessing..."

# Run the data cleaning script
echo "Step 1: Running data cleaning script..."
python /cluster/home/pettdalh/tdt4265_project/preprocessing/clean_data.py

# Check if the first script executed successfully
if [ $? -eq 0 ]; then
    echo "Data cleaning completed successfully."
else
    echo "Data cleaning failed! Exiting..."
    exit 1
fi

# Run the data augmentation script
echo "Step 2: Running data augmentation script..."
python /cluster/home/pettdalh/tdt4265_project/preprocessing/augment_data.py

# Check if the second script executed successfully
if [ $? -eq 0 ]; then
    echo "Data augmentation completed successfully."
else
    echo "Data augmentation failed!"
    exit 1
fi

echo "Data preprocessing completed successfully!"