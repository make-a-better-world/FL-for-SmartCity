#!/bin/bash

for i in `seq 0 19`; do
    p_id=$(( i % 10))
    echo "Starting client $i $p_id"
    python3 client.py --config="config.yaml" --partition=${p_id} --client_id=${i} &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
