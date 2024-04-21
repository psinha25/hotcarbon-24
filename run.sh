#!/bin/bash -e

log() {
    echo -e "$@"
    if [[ ! -z ${PRINT_OUTS} ]]; then
        echo -e "$@" >> ${PRINT_OUTS}
    fi
}

print_log_location() {
    local run_sh_exit_code=$?
    log "Exiting with error_code=${run_sh_exit_code} (0 is clean exit)"
    log "Examine ${PRINT_OUTS} for logs"
    exit ${run_sh_exit_code}
}

print_help() {
    log "Usage: ${0} [OPTIONS] model1-parameter model2-paramete ..."
    log "Options:"
    log "  --device-type   DEVICE_TYPE                   v100, a100, h100                                   (required)"
    log "  --device-id     DEVICE_ID                     0, 1, 2, ..                                        (required)"
    log "  --modes         MODE1,MODE2,MODE3             mps-uncap,tm                                       (default mps-uncap,tm)"
    log "  --duration      DURATION_OF_EXPR_IN_SECONDS"
    log "  -h, --help                                    Show this help message"
    log -e "\n"

    log "Examples:"
    log " $0 --device-type v100 --device-id 0 --duration 10 diffusion-1"
    log " $0 --device-type v100 --device-id 1 --duration 20 diffusion-1 whisper-1"
    log -e "\n"

    echo "NOTE: Only support closed loop and TM right now. MPS support in progress"
}

get_input() {
    model_run_params=()
    duration=120
    modes=("mps-uncap" "tm")
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --device-type)
                device_type="$2"
                shift 2
                ;;
            --device-id)
                device_id="$2"
                shift 2
                ;;
            --duration)
                duration=$2
                shift 2
                ;;
            --modes)
                unset modes
                IFS=',' read -r -a modes <<< "$2"
                shift 2
                ;;
            -h|--help)
                print_help
                exit 0
                ;;
            *)
                model_run_params+=("$1")
                shift
                ;;
        esac
    done
}

validate_input() {
    num_procs=${#model_run_params[@]}
    if [[ -z ${device_type} || (${device_type} != "4090" && ${device_type} != "a100" && ${device_type} != "a6000") ]]; then
        log "Invalid device_type: ${device_type}"
        print_help
        exit 1
    fi

    if [[ -z ${device_id} || ! ${device_id} =~ ^[0-9]+$ ]]; then
        log "Invalid device_id: ${device_id}"
        print_help
        exit 1
    fi

    if [[ ${num_procs} -eq 0 ]]; then
        log "Need at least 1 model configuration to run"
        print_help
        exit 1
    fi

    for mode in ${modes[@]}
    do
        if [[ ${mode} != "tm" && ${mode} != "mps-uncap" ]]; then
            log "Invalid mode: ${mode}"
            log "Must be one of: tm"
            print_help
            exit 1
        fi
    done
}

parse_input() {
    pattern="^[^-]+-[^-]+$"
    models=()
    models_and_batch_sizes=()
    batch_sizes=()
    for (( i=0; i<${num_procs}; i++ )); do
        element=${model_run_params[$i]}
        if [[ ! "${element}" =~ $pattern ]]; then
            log "Expected: Model-BatchSize"
            log "Got: ${element}"
            log "Example: diffusion-1"
            exit 1
        fi
        models[$i]=$(echo $element | cut -d'-' -f1)
        batch_sizes[$i]=$(echo $element | cut -d'-' -f2)
        models_and_batch_sizes[$i]=${models[$i]}"-"${batch_sizes[$i]}
    done
}

setup_expr() {
    source helper.sh && helper_setup
    WS=$(git rev-parse --show-toplevel)
    get_result_dir models[@] batch_sizes[@] ${device_type}
    trap print_log_location EXIT
    PRINT_OUTS=/tmp/print_outs-$(uuidgen | cut -c 1-8).txt
    rm -f ${PRINT_OUTS}
    echo "Logs at ${PRINT_OUTS}"
}

run_cmd() {
    local cmd_arg="$1"
    local device_id_arg=$2
    local mode_arg=$3
    local duration_arg=$4
    local run_id_arg=$5

    echo "Running: ${cmd_arg}" >> ${PRINT_OUTS}
    eval "${cmd_arg} >> ${PRINT_OUTS} 2>&1 &"
    run_expr_pid=$!

    # Wait for the pipe to be created
    pipe=/tmp/${run_id_arg}
    local ctr=0
    while [[ ! -p ${pipe} && $ctr -lt 100 ]]; do
        sleep 0.01
        ctr=$((ctr+1))
    done
    sleep 1

    # Start the experiment
    timeout 1 bash -c "echo '{\"device-id\": ${device_id_arg}, \"mode\": \"${mode_arg}\"}' > ${pipe}"

    # Wait for experiment to start and all models to be loaded
    load_ctr=10000
    while [[ ${load_ctr} -gt 0 ]]; do
        if [[ -f /tmp/${run_id_arg}.load ]]; then
            break
        fi

        if ! kill -0 "${run_expr_pid}" &> /dev/null; then
            echo "Process no longer alive"
            exit 1
        fi

        ((load_ctr--))
        sleep 0.25
    done

    # Wait for duration_arg
    local ctr=0
    while :
    do
        sleep 1
        ctr=$((ctr+1))

        # Waiting for sleep duration
        if [[ ${ctr} -eq ${duration_arg} ]]; then
            break
        fi

        # If the experiment dies before that => then error
        if ! taskset -c 0 kill -0 ${run_expr_pid} 2>/dev/null; then
            exit 1
        fi
    done

    # Stop the experiment
    timeout 1 bash -c "echo '{\"mode\": \"stop\"}' > ${pipe}"

    # Wait for the process to exit
    wait ${run_expr_pid}
    run_expr_exit_code=$?
    if [[ ${run_expr_exit_code} -ne 0 ]]; then
        exit ${run_expt_exit_code}
    fi
    echo -e "===============================\n\n" >> ${PRINT_OUTS}
}

generate_json_input() {
    export model="\"$1\""
    export batch_size=$2

    template=$(cat model-param.json.template)
    json_input=$(echo $template | envsubst)

    unset model batch_size
}

generate_model_params() {
    declare -a json_arr=("${!1}")
    model_params='[]'
    for json_str in "${json_arr[@]}"; do
        model_params=$(jq ". += [$json_str]" <<< "$model_params")
    done
}

generate_closed_loop_load() {
    json_array=()
    for (( i=0; i<${num_procs}; i++ ))
    do
        generate_json_input ${models[$i]} ${batch_sizes[$i]} 0
        json_array+=("${json_input}")
    done

    generate_model_params json_array[@]

    for mode in ${modes[@]}
    do
        local run_id_arg=$(uuidgen)
        cmd="./run_job_mix.sh --device-type ${device_type} --run-id ${run_id_arg} '${model_params}'"
        log "Running closed loop experiment for ${mode}"
        run_cmd "${cmd}" ${device_id} ${mode} ${duration} ${run_id_arg}
    done


}

get_input $@
validate_input
parse_input
setup_expr
generate_closed_loop_load
log "Run success: results are stored in ${result_dir}"
