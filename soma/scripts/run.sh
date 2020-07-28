#!/bin/bash
for i in {20..100}
do
    echo "Running seed $i"
    python3 soma_sim_model_trainer_sink.py $i
    python3 soma_sim_model_trainer_tcn.py $i
done
