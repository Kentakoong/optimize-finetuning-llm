# Optimize LLM Fine-tuning for LANTA

## Installation

### Install using Mamba

```bash
mamba create -n <your-env-name> python=3.10 # or -p <your-env-path> python=3.10

mamba activate <your-env-path>

cd <base-path-to-repo>

bash shell/install-packages.sh --env_path <your-env-path>
```

## Edit the `submit-batch.sh` file

Edit the `submit-batch.sh` file to set your account

from `#SBATCH -A xxyyyyyy` to `#SBATCH -A <your-credit-account>`

## Submit Train Model

```bash
sbatch -N <node-count> submit-batch.sh
```

### Setting Default Options

To set the default options, you can create the `pre-submit.sh` file in the base directory of the repository.

These are the required options that needs to be set before submitting the job.

```bash
export PROJ_PATH=<base-path-to-repo>
export SHARED_PATH=<your-shared-path>
export CACHE_PATH=<your-cache-path>
export ENV_PATH=<your-env-path>

# and export more options below...
```

### Options

The following options are used for `arguments` and `pre-submit file` for when using arguments with the script.
(N/A means that the option isn't available for either `arguments` or `pre-submit file`.)

- `--nthreads` | `NTHREADS`: Specifies the number of CPU helper threads used per network connection for socket transport. (default: `8`)
- `--pthreads` | `PTHREADS`: Specifies the number of sockets opened by each helper thread of the socket transport. (default: `2`)
- `--batch_size` | `BATCH_SIZE`: Specifies the batch size. (default: `4`)
- `--deepspeed_stage` | `DEEPSPEED_STAGE`: Specifies the deepspeed stage. (default: `2`)
- `--model_size` | `MODEL_SIZE`: Specifies the model size. (default: `7b`)
- `--task` | `TASK`: Specifies the task. (default: `default`)
  **Options are:**
  - `nccl`: To name the log folder as NCCL testing structure (`NTHREADS`nth-`PTHREADS`pth-`SLURM_JOB_ID`)
  - `scaling`: To name the log folder for scaling (`STAGE`-`MODEL_SIZE`/`COUNT_NODE`n-`BATCH_SIZE`b-`SLURM_JOB_ID`)
  - `default`: To name the log folder as default (`COUNT_NODE`n-`BATCH_SIZE`b-`SLURM_JOB_ID`)
- `--run_with` | `RUN_WITH`: Specifies the environment to run the script. (default: `conda`)
- `N/A` | `BASE_PATH`: Specifies the base path to the repository.
- `N/A` | `SHARED_PATH`: Specifies the shared path.
- `N/A` | `CACHE_PATH`: Specifies the cache path.
- `--env_path` | `ENV_PATH`: Specifies the path to the environment.
