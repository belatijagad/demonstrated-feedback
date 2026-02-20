#!/bin/bash

SEED=42
seed_file="/tmp/seed_$SEED"

echo "$SEED" | openssl sha256 -binary > "$seed_file"

selected_numbers=$(shuf -i 1-50 -n 12 --random-source="$seed_file")

echo "Seed: $SEED"
echo "Selected indices: $selected_numbers"

benchmark="ccat50"

for num in $selected_numbers; do
    echo "Running experiment for author_key: $num"
    
    python scripts/run_ditto.py configs/ditto-mistral-7b-instruct.yaml \
        --train_pkl=benchmarks/${benchmark}/processed/${benchmark}_train.pkl \
        --train_author_key=$num \
        --output_dir=outputs/${benchmark}-mistral-7b-instruct-ditto-key-$num \
        --train_instances=7
done

rm "$seed_file"