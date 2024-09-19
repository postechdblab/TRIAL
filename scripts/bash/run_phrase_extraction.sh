#!/bin/bash

# Define the total variable
begin=0
end=35
total=144
num_devices=8  # Number of available devices (0 to num_devices-1)

# Print the command being executed
echo "Running: python scripts/preprocess/extract_phrases.py +total=$total +op=split_file +indices=[$begin,$end]"
python scripts/preprocess/extract_phrases.py +total=$total +op=split_file +indices=[$begin,$end]

# Loop from 0 to total-1
for i in $(seq $begin $end); do
    # Calculate the device to use (round-robin distribution)
    device=$((i % num_devices))

    # Print the command being executed
    echo "Running: CUDA_VISIBLE_DEVICES=$device python scripts/preprocess/extract_phrases.py +total=$total +i=$i +op=extract"

    # Run the command with the assigned device and store the output in a text file
    CUDA_VISIBLE_DEVICES=$device python scripts/preprocess/extract_phrases.py +total=$total +i=$i +op=extract > output_$i.txt 2>&1 &
done

# Wait for all background processes to complete
wait

echo "All done!"