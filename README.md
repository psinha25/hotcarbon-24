## Environment Setup

Setup using Python virtual environment
```
[sudo] apt install python3-venv
python3 -m venv ~/virtual/hotcarbon
source ~/virtual/hotcarbon/bin/activate
pip3 install -r packaging/requirements.txt
```

## How to Run Experiments

```
export USE_SUDO=1   # If you want to run MPS

# Use python virtual environment
export VENV=~/virtual/hotcarbon

# To get how to use it 
./run.sh --help

# Example
./run.sh --device-type a100 --device-id 1 --modes tm --duration 20 diffusion-1 whisper-1
```

`run.sh` takes a few arguments:
- `device-type`: 4090, a100, a6000
- `device-id`: 0 - 5
- `modes`: tm (time multipliexing), mps-uncap (MPS)
- `duration`: in seconds, how long experiments runs after models are loaded

The final arguments are the set of models and batch sizes you want to run on a single GPU. In the example above, we want to co-locatea diffusion model and whisper speech recognition model on the same A100 GPU, each with a batch size 1, using time multiplexing. The model format is <model_name>-<batch_size> (e.g., diffusion-1). 

For now, only batch size 1 is supported. We support four models currently. The names of you pass should be the following:

- diffusion
- whisper
- bert
- gpt

After `run.sh` completes (depends on how long you set the `--duration` argument to), you will see an output like below:
```
Logs at /tmp/print_outs-bb667276.txt
Running closed loop experiment for tm
Run success: results are stored in results/4090/diffusion-whisper/diffusion-1_whisper-1
Exiting with error_code=0 (0 is clean exit)
Examine /tmp/print_outs-bb667276.txt for logs
```
If the `error_code=0`, everything ran successfully. You can check the logs at the specific log file path. The output also contains the location in the git directory where the results are stored. There will be 7 files generated:
- `tput.csv` - throughput (requests/second) for each model
- `total_p0.csv` - minimum latency of a single request
- `total_p50.csv` - median latency of a single request
- `total_90.csv` - p90 latency of a single request
- `total_99.csv` - p99 latency of a single request
- `total_100.csv` - max latency of a single request
- `pwr.csv` - power data we collect over course of experiment