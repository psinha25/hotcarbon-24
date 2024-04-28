#!/bin/bash

source ../helper.sh

PYTHON=${VENV}/bin/python3
NCU=/usr/local/NVIDIA-Nsight-Compute/ncu

run_decorated_inference() {
    pre=$1
    post="> /dev/null 2>&1"
    if [[ $# -eq 2 ]]; then
        post=$2
    fi

    command="${pre} \
        ${PYTHON} ../src/executor.py \
        --device-id ${device_id} \
        --model ${model} \
        --batch-size ${batch} \
        --num-infer 1 \
        --tid 0 \
        ${post} &"
    
    # echo $command
    eval "$command"
    ncu_pid=$
    sleep 10

    readarray -t forked_pids < <(ps -eaf | grep executor.py | grep -v "${NCU}" | grep -v "nsys" | grep -v "nsight" | grep -v grep | awk '{print $2}')
    if [[ ${#forked_pids[@]} != 1 ]]; then
        echo "Expected 1 executor.py process! Seen != 1..."
        echo "Inspect using command: ' ps -eaf | grep executor.py | grep -v "${NCU}" | grep -v "nsys" | grep -v grep'"
        return 1
    fi

    # Wait for the model to load
    for pid in "${forked_pids[@]}"
    do
        while :
        do
            if [[ -f /tmp/${pid} ]]; then
                lt="$lt, $(cat /tmp/${pid})"
                ${SUDO} rm -f /tmp/${pid}
                loaded_procs+=(${pid})
                break
            elif [[ -f /tmp/${pid}_oom ]]; then
                ${SUDO} rm -f /tmp/${pid}_oom
                break
            fi

        done
    done

    # Start inference
    ${SUDO} kill -SIGUSR1 ${forked_pids[@]}

    # Wait till the prefixed command completes
    while ${SUDO} kill -0 ${ncu_pid} >/dev/null 2>&1; do sleep 1; done

}

profile_model() {
    device_type=$1
    device_id=$2
    model=$3
    batch=$4
    model_type=$5

    if [[ (${device_type} != "4090" && ${device_type} != "a100" && ${device_type} != "a6000") ]]; then
        echo "Invalid device_type: ${device_type}"
        print_help
        exit 1
    fi

    if [[ -z ${device_id} || ! ${device_id} =~ ^[0-9]+$ ]]; then
        echo "Invalid device_id: ${device_id}"
        print_help
        exit 1
    fi

    result_dir=$(pwd)/data/${device_type}/${model}

    mkdir -p ${result_dir}

    echo 'NCU CSV...'
    run_decorated_inference "${SUDO} -E ${NCU} -f --csv --set detailed --nvtx --nvtx-include \"start/\"" "> ${result_dir}/batchsize_${batch}_output_ncu.csv"

    echo 'NSYS Profile...'
    run_decorated_inference "${SUDO} -E nsys profile --show-output true --sample none --trace cuda,nvtx,osrt,cudnn,cublas --stats=true --force-overwrite true --stop-on-exit true --wait=all -o ${result_dir}/batchsize_${batch}_output_nsys --capture-range=cudaProfilerApi --cudabacktrace=true --gpu-metrics-device=0" 

    echo 'NSYS CONVERSION TO CSV...'
    ${SUDO} -E nsys stats --report gputrace --format csv,column --force-export=true --output ${result_dir}/batchsize_${batch}_output_nsys,- ${result_dir}/batchsize_${batch}_output_nsys.nsys-rep
    sed -i '/^"ID","Process ID"/,$!d' ${result_dir}/batchsize_${batch}_output_ncu.csv
}

if [[ $# -lt 4 ]]; then
    echo "Expected Syntax: '$0 <4090 | a100 | a6000> <device_id> model batchsize"
    echo "Examples:"
    echo " $0 v100 0 vgg11 16 vision"
    exit 1
fi

profile_model $@
