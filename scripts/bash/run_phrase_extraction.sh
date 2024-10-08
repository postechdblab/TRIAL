#!/bin/bash

# Define the total variable
begin=128
end=147
total=196
num_devices=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# Create the outputs directory if it doesn't exist
output_dir="outputs"
if [ ! -d "$output_dir" ]; then
  mkdir -p "$output_dir"
  echo "Created directory: $output_dir"
fi

# Print the command being executed
echo "Number of GPU devices: $num_devices"
echo "Running: python scripts/preprocess/extract_phrases.py +total=$total +op=split_file +indices=[$begin,$end]"
python scripts/preprocess/extract_phrases.py +total=$total +op=split_file +indices=[$begin,$end]

# Loop from 0 to total-1
for i in $(seq $begin $end); do
    # Calculate the device to use (round-robin distribution)
    device=$((i % num_devices))

    # Print the command being executed
    echo "Running: CUDA_VISIBLE_DEVICES=$device python scripts/preprocess/extract_phrases.py +total=$total +i=$i +op=extract"

    # Run the command with the assigned device and store the output in a text file
    CUDA_VISIBLE_DEVICES=$device python scripts/preprocess/extract_phrases.py +total=$total +i=$i +op=extract > "${output_dir}/output_$i.txt" 2>&1 &
done

# Wait for all background processes to complete
wait

echo "All done!"