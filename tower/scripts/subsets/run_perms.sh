#!/bin/bash
for i in {10..1920..50}
do
    echo "Running demos $i"
    python3 Tower_building_tcn_hung_sequencing_generalisation_subsets.py $i
 #   python3 Tower_building_sink_sequencing_generalisation_subsets.py $i
done
