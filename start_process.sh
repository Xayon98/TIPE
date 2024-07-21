#!/bin/bash

n=$1  # Take n as a parameter from argv

for ((id=0; id<n; id++))
do
    nohup python3 -B -u process_annealing.py ${id} &> "annealing${id}.out" &
done