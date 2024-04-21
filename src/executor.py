import argparse
import signal
import time
import sys
import pickle
import torch
import os
from inference import get_inference_object # type: ignore


WARMUP_REQS = 5
LARGE_NUM_REQS = 100000

class InferenceExecutor:
    def __init__(self, model_obj, num_infer, tid):
        self.model_obj = model_obj
        self.num_infer = num_infer
        self.tid = tid

        # Process synchronization mechanism
        self.start = False
        self.finish = False
        self.job_completed = False
    
    def _catch_to_start(self, signum, frame):
        self.start = True

    def _catch_to_end(self, signum, frame):
        self.finish = True
        self.job_completed = True
    
    def _indicate_ready(self):
        # Wait til user instructs to start via signal handler
        self.install_signal_handler()
        
        # Write to indicate readiness
        # there by the user can signal when all procs are ready
        with open(f"/tmp/{os.getpid()}", "w") as ready_file:
            ready_file.write("")
        while not self.start:
            pass
        return

    def _return_infer_stats(self, infer_stats):
        infer_stats.insert(0, f"{self.model_obj.get_id()}")
        result = (self.tid, infer_stats)
        with open(f"/tmp/{os.getpid()}.pkl", "wb") as h:
            pickle.dump(result, h)

    def install_signal_handler(self):
        signal.signal(signal.SIGUSR1, self._catch_to_start)
        signal.signal(signal.SIGUSR2, self._catch_to_end)

    def run_infer_executor(self, num_reqs):
        completed = 0
        total_time_arr = []

        process_start_time = time.time()
        for _ in range(num_reqs):
            if self.job_completed:
                break
            
            start_time = time.time()
            completed += self.model_obj.infer()
            end_time = time.time()
            
            total_time = end_time - start_time
            total_time_arr.append(total_time)

        process_end_time = time.time()

        return [
            completed / (process_end_time - process_start_time),
            total_time_arr
        ]

    def run(self):
        # Load Model and transfer inputs
        self.model_obj.load_model()
        self.model_obj.load_data()

        # Warm up the model
        self.run_infer_executor(WARMUP_REQS)
        reqs_completed = WARMUP_REQS

        # Ready for experiment
        self._indicate_ready()

        # Start experiment
        if torch.cuda.is_available():
            torch.cuda.cudart().cudaProfilerStart()
            torch.cuda.nvtx.range_push("start")
        infer_stats = self.run_infer_executor(
            self.num_infer,
        )
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_pop()
            torch.cuda.cudart().cudaProfilerStop()
        
        # Give stats back to user
        self._return_infer_stats(infer_stats)


if __name__=='__main__':

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--model", type=str, default='diffusion')
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-infer", type=int, default=sys.maxsize)
    parser.add_argument("--tid", type=int, default=0)
    opt, unused_args = parser.parse_known_args()

    # Create batched inference object
    model_obj = get_inference_object(
        opt.model,
        opt.device_id,
        opt.batch_size
    )

    executor = InferenceExecutor(
        model_obj,
        opt.num_infer,
        opt.tid
    )

    executor.run()


    # model_obj.load_model()
    # model_obj.load_data()
    # size = model_obj.infer()
    # print(size)
