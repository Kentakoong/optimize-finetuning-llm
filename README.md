# Optimize LLM Fine-tuning for LANTA

## Installation

### Install using Mamba

```bash
mamba create -n lanta-llm python=3.10

mamba activate lanta-llm

cd <base-path-to-repo>

bash shell/install-packages.sh
```

## Submit Train Model

```bash
sbatch submit-batch.sh
```

### Options

- `-N`: Specifies the number of nodes.
- `--nthreads`: Specifies the number of CPU helper threads used per network connection for socket transport. (default: `8`)
- `--pthreads`: Specifies the number of sockets opened by each helper thread of the socket transport. (default: `2`)
- `--batch_size`: Specifies the batch size. (default: `4`)
- `--deepspeed_stage`: Specifies the deepspeed stage. (default: `2`)
- `--model_size`: Specifies the model size. (default: `7b`)
- `--task`: Specifies the task. (default: `default`)
  
  **Options are:**
  - `nccl`: To name the log folder as NCCL testing structure (`NTHREADS`nth-`PTHREADS`pth-`SLURM_JOB_ID`)
  - `scaling`: To name the log folder for scaling (`STAGE`-`MODEL_SIZE`/`COUNT_NODE`n-`BATCH_SIZE`b-`SLURM_JOB_ID`)
  - `default`: To name the log folder as default (`COUNT_NODE`n-`BATCH_SIZE`b-`SLURM_JOB_ID`)
- `--run_with`: Specifies the environment to run the script. (default: `conda`)
- `--env_path`: Specifies the path to the environment.
