#!/bin/bash
for i in {1..20}
do
    echo "Running seed $i"
    python3 Tower_building_sink_sequencing.py $i
#    python3 Tower_building_bc_sequencing.py $i
#    python3 Tower_building_tcn_sequencing.py $i
done
