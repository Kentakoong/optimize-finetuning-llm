echo "Moving old logs"

mv ./logs/* ../logs

echo "Done moving old logs"

sbatch submit-batch.sh
