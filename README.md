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

From:  `#SBATCH -A xxyyyyyy`

To:&nbsp;&nbsp;&nbsp;&nbsp; `#SBATCH -A <your-credit-account>`

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

| `Arguments`         | `Pre-Submit file` | `Default Value` | `Description`                                                                                |
| ------------------- | ----------------- | --------------- | -------------------------------------------------------------------------------------------- |
| `--nthreads`        | `NTHREADS`        | `8`             | Specifies the number of CPU helper threads used per network connection for socket transport. |
| `--pthreads`        | `PTHREADS`        | `2`             | Specifies the number of sockets opened by each helper thread of the socket transport.        |
| `--batch_size`      | `BATCH_SIZE`      | `4`             | Specifies the batch size.                                                                    |
| `--deepspeed_stage` | `DEEPSPEED_STAGE` | `2`             | Specifies the deepspeed stage.                                                               |
| `--model_size`      | `MODEL_SIZE`      | `7b`            | Specifies the model size.                                                                    |
| `--task`            | `TASK`            | `default`       | Specifies the task. [More Details](#task-options)                                            |
| `--run_with`        | `RUN_WITH`        | `conda`         | Specifies the environment to run the script.                                                 |
| `N/A`               | `PROJ_PATH`       |                 | Specifies the path to the repository.                                                        |
| `N/A`               | `SHARED_PATH`     |                 | Specifies the shared path.                                                                   |
| `N/A`               | `CACHE_PATH`      |                 | Specifies the cache path.                                                                    |
| `--env_path`        | `ENV_PATH`        |                 | Specifies the path to the environment.                                                       |

#### Task Options

**Options are:**

- `nccl`: To name the log folder as NCCL testing structure
  - `NTHREADS`nth-`PTHREADS`pth-`SLURM_JOB_ID`
- `scaling`: To name the log folder for scaling
  - `STAGE`-`MODEL_SIZE`/`COUNT_NODE`n-`BATCH_SIZE`b-`SLURM_JOB_ID`
- `default`: To name the log folder as default
  - `COUNT_NODE`n-`BATCH_SIZE`b-`SLURM_JOB_ID`
