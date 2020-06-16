#!/bin/bash
for i in {1..720..50}
do
    echo "Running demos $i"
    python3 Tower_building_sink_sequencing_generalisation_unique.py $i

done
