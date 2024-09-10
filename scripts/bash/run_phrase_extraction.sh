#!/bin/bash

# Define the total variable
begin=0
end=48
total=48
num_devices=5  # Number of available devices (0 to 4)

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