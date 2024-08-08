mv ./logs/* ./logsed/
sbatch -N 2 submit-batch-wo-node.sh --nthreads 8 --pthreads 2
