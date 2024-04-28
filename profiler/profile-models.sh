#!/bin/bash

# Check if device type is specified as a command line argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <device_type>"
    echo "device_type must be one of: 4090, a100, a6000"
    exit 1
fi

device_type=$1

# Check if device_type is one of the allowed options
case $device_type in
    4090|a100|a6000) ;;
    *)
        echo "Invalid device_type. Must be one of: 4090, a100, a6000"
        exit 1
        ;;
esac

models=('bert' 'diffusion' 'gpt' 'whisper')
batches=(1)

for model in "${models[@]}"; do
    for batch in "${batches[@]}"; do
        command="./profiler.sh $device_type 0 $model $batch"
        echo "Profiling ${model} with batch size ${batch}"
        eval "$command"
        echo "--------------------------------------------"
        echo ""
    done
done

# for model in "${models[@]}"; do
#     for batch in "${batches[@]}"; do
#         command="python3 extract_profile.py --model ${model} --batch_size ${batch} --profile_dir ./data/a100/${model}/"
#         echo "Aggregating profile for ${model} with batch size ${batch}"
#         eval "$command"
#     done
# done