conda activate ditto

export HF_TOKEN=""
export PYTHONPATH=$PYTHONPATH:.

benchmark="ccat50"

rm -rf outputs/${benchmark}-mistral-7b-instruct-ditto

python scripts/run_ditto.py configs/ditto-mistral-7b-instruct.yaml \
    --train_pkl=benchmarks/${benchmark}/processed/${benchmark}_train.pkl \
    --train_author_key=0 \
    --output_dir=outputs/${benchmark}-mistral-7b-instruct-ditto \
    --train_instances=7

# python generate.py \
#     --benchmark=$benchmark \
#     --train_author_key=0

