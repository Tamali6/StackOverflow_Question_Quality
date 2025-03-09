#!/bin/bash

echo "Starting Stackoverflow question classification..."

# Train the model using config.yaml
echo "Training the model..."
python src/train.py  --config config/config.yaml

# Test the model
echo "Evaluating the model..."
python src/test.py   --config config/config.yaml

echo "Execution completed."
