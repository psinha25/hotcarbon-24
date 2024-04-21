#!/bin/bash -e

print_help() {
    echo "Usage: ${0} [OPTIONS] model1-parameter model2-parameter ..."
    echo "Options:"
    echo "--device-type     DEVICE_TYPE     4090, a100, a6000"
    echo "--run-id          UNIQUE RUN ID   uuid"
    echo "-h, --help                        Show this help message"
    echo -e "\n"

    echo "Example"
    echo " $0 --device-type a100 --load 1 '[{\"model\": \"diffusion\", \"batch-size\": 2}, {\"model\": \"whisper\", \"batch-size\": 4}]'"
    echo -e "\n"

    echo "NOTE: Only support closed loop and TM right now. MPS support in progress"
}

get_input() {
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --device-type)
                device_type="$2"
                shift 2
                ;;
            --run-id)
                run_id="$2"
                shift 2
                ;;
            -h|--help)
                print_help
                exit 0
                ;;
            *)
                parse_model_parameters "$@"
                shift $#
            ;;
        esac
    done
    num_procs=${#model_run_params[@]}
}

validate_input() {
    if [[ (${device_type} != "4090" && ${device_type} != "a100" && ${device_type} != "a6000") ]]; then
        echo "Invalid device_type: ${device_type}"
        print_help
        exit 1
    fi

    if [[ -z ${run_id} ]]; then
        echo "run-id is a required argument"
        print_help
        exit 1
    fi

    if [[ ${num_procs} -eq 0 ]]; then
        echo "Need at least 1 model configuration to run"
        print_help
        exit 1
    fi
}

cleanup_handler() {
    original_x=$(set +o | grep xtrace)
    set -x
    local job_mix_exit_code=$?

    # Kill any pending procs
    if [[ ${#uuids_ran[@]} -gt 0 ]]; then
        IFS="|"
        uuid_grep="${uuids_ran[*]}"
        unset IFS
        ps -eaf | grep batched_inference_executor.py | egrep "${uuid_grep}" |
            grep -v grep | awk '{print $2}' |
            xargs -I{} kill -9 {} || :
    fi

    # Clean the modes ran
    for ((y=0; y<${#modes_ran[@]}; y++))
    do
        echo "Cleaning up for ${modes_ran[$y]} on ${device_ids_ran[$y]}"
        cleanup ${modes_ran[$y]} ${device_ids_ran[$y]} || :
    done

    # Clean up the fifo pipe created
    rm -f ${fifo_pipe} || :

    # Clean up the IPC queue
    ipcrm --all=msg || :

    # Exit with the exit code with which the handler was called
    echo "Exiting with Error code: ${job_mix_exit_code} (0 is clean exit)"
    eval "$original_x"
    exit ${job_mix_exit_code}
}

setup_expr() {
    trap cleanup_handler EXIT

    # Find where to store results
    get_result_dir models[@] batch_sizes[@] ${device_type}
    mkdir -p ${result_dir}

    # Create a FIFO to listen on
    fifo_pipe=/tmp/${run_id}
    rm -f ${fifo_pipe}
    mkfifo ${fifo_pipe}

    # Clean up the IPC queue
    ipcrm --all=msg || :
}

read_fifo() {
    pipe_name=$1
    read json_data < ${pipe_name} # blocking
    echo "Got: ${json_data}"
    mode_to_run=$(echo "$json_data" | jq -r '.mode')
    device_id_to_run=$(echo "$json_data" | jq -r '.["device-id"]')
    modes_ran+=(${mode_to_run})
    device_ids_ran+=(${device_id_to_run})
}

start_expr() {
    local mode_arg=$1
    local device_id_arg=$2
    local uuid_arg=$3
    local run_id_arg=$4

    enable_mps_if_needed ${mode_arg} ${device_id_arg}

    cmd_arr=()
    for (( c=0; c<${num_procs}; c++ )); do
        if [[ ${mode_arg} == "mps-uncap" ]]; then
            export_prefix="export CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING=0 && \
                           export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_${device_id_arg} && \
                           export CUDA_VISIBLE_DEVICES=0"
        else
            export_prefix="export CUDA_VISIBLE_DEVICES=${device_id_arg}"
        fi

        # Assumes: we can run 7 models in parallel in a device
        cpu=$(((device_id_arg * 7) + (c+1)))

        # We always set device-id to 0, as CUDA_VISIBLE_DEVICES exports only 1 GPU per model
        cmd="${export_prefix} && \
            taskset -c ${cpu} python3 src/executor.py \
            --device-id 0 \
            ${model_run_params[$c]} \
            --run-id ${run_id_arg} \
            --tid ${c} \
            --uuid ${uuid_arg} > /dev/null &"

        eval $cmd
        cmd_arr+=("${cmd}")
    done

    # Check if processes are alive
    readarray -t forked_pids < <(ps -eaf | grep batched_inference_executor.py |
        grep ${uuid_arg} | grep -v grep |
        awk '{for (i=1; i<=NF; i++) if ($i == "--tid") print $(i+1),$0}' |
        sort -n | cut -d' ' -f2- | awk '{print $2}')
    if [[ ${#forked_pids[@]} -ne ${num_procs} ]]; then
        echo "Expected ${num_procs} processes. But found ${#forked_pids[@]}}"
        echo "Examine commands: "
        for cmd in "${cmd_arr[@]}"
        do
            echo "  ${cmd}"
        done
        exit 1
    fi

    # Wait till all pids have loaded their models
    lt=""
    loaded_procs=()
    echo ${forked_pids[@]}
    for pid in "${forked_pids[@]}"
    do
        load_ctr=1000
        while [[ ${load_ctr} -gt 0 ]];
        do
            if [[ -f /tmp/${pid} ]]; then
                lt="$lt, $(cat /tmp/${pid})"
                rm -f /tmp/${pid}
                loaded_procs+=(${pid})
                break
            fi

            if ! kill -0 "${pid}" &> /dev/null; then
                echo "Process no longer alive"
                load_ctr=0
                break
            fi
            ((load_ctr--))
            sleep 0.25
        done

        if [[ ${load_ctr} -eq 0 ]]; then
            echo "Some of the models did not load!"
            echo "Examine commands: "
            for cmd in "${cmd_arr[@]}"
            do
                echo "  ${cmd}"
            done
            exit 1
        fi
    done

    # Touch to indicate models are loaded
    touch /tmp/${run_id_arg}.load

    # Start inference
    echo "Starting inference on ${loaded_procs[@]}"
    kill -SIGUSR1 ${loaded_procs[@]}

}

run_expr() {

    local procs=()
    local prev=()

    read_fifo ${fifo_pipe}
    while [[ ${mode_to_run} != "stop" ]]; do
        # We could enable persistence mode here -- not sure given we are doing power readings
        #  ${SUDO} nvidia-smi -i ${device_id_to_run} -pm ENABLED

        # acquire lock on the GPU
        lock_gpu ${device_id_to_run}

        # Get a unique id for the run
        local uuid=$(uuidgen)
        uuids_ran+=(${uuid})

        # Begin experiment
        start_expr ${mode_to_run} ${device_id_to_run} ${uuid} ${run_id}

        # Keep track of state
        prev=("${loaded_procs[@]}")
        procs+=( "${loaded_procs[@]}" )
        prev_mode_run=${mode_to_run}
        prev_device_id_run=${device_id_to_run}

        # Blocking until the next command from control plane comes in
        read_fifo ${fifo_pipe}
    done

    # Make sure to stop all inferences
    safe_clean_gpu prev[@] ${prev_mode_run} ${prev_device_id_run}

    # Wait for the process to exit
    for pid in "${procs[@]}"
    do
        while taskset -c 0 kill -0 ${pid} >/dev/null 2>&1; do sleep 1; done
    done

}   

git_dir=$(git rev-parse --show-toplevel)
cd ${git_dir}
source helper.sh && helper_setup
get_input $@
validate_input
setup_expr
run_expr