echo "Moving old logs"

mv ./logs/* ./logsed

echo "Done moving old logs"

sbatch submit-batch.sh
