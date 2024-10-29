#!/bin/bash

# Define the total variable
begin=0
end=1
total=2
num_devices=2  # Number of available devices (0 to num_devices-1)
dataset_name="beir-arguana"

# Create the outputs directory if it doesn't exist
output_dir="outputs"
if [ ! -d "$output_dir" ]; then
  mkdir -p "$output_dir"
  echo "Created directory: $output_dir"
fi

# Print the command being executed
echo "Running: python scripts/preprocess/tokenize_data.py dataset.name=$dataset_name +total=$total +op=split_file +indices=[$begin,$end]"
python scripts/preprocess/tokenize_data.py dataset.name=$dataset_name +total=$total +op=split_file +indices=[$begin,$end]

# Loop from 0 to total-1
for i in $(seq $begin $end); do
    # Calculate the device to use (round-robin distribution)
    device=$((i % num_devices))

    # Print the command being executed
    echo "Running: CUDA_VISIBLE_DEVICES=$device python scripts/preprocess/tokenize_data.py dataset.name=$dataset_name +total=$total +i=$i +op=tokenize"

    # Run the command with the assigned device and store the output in a text file
    CUDA_VISIBLE_DEVICES=$device python scripts/preprocess/tokenize_data.py dataset.name=$dataset_name +total=$total +i=$i +op=tokenize > ${output_dir}/output_$i.txt 2>&1 &
done

# Wait for all background processes to complete
wait

echo "All done!"