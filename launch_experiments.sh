#!/bin/bash -e

mixes=(
# "diffusion-1"
"whisper-1"
"bert-1"
"gpt-1"
"bert-1 whisper-1"
"bert-1 gpt-1"
"bert-1 diffusion-1"
"whisper-1 gpt-1"
"whisper-1 diffusion-1"
"gpt-1 diffusion-1"
"bert-1 whisper-1 gpt-1"
"bert-1 whisper-1 diffusion-1"
"bert-1 gpt-1 diffusion-1"
"whisper-1 gpt-1 diffusion-1"
)

for ((i=0; i<${#mixes[@]}; i++)); do
    mix="${mixes[$i]}"
    cmd="./run.sh --device-type a100 --device-id 0 --modes tm --duration 300 ${mix}"
    echo "${cmd}"
    eval $cmd
done
