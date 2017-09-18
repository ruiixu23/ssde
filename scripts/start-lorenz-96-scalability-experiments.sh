#!/usr/bin/env bash

module load new gcc/4.8.2 openblas/0.2.13_seq sqlite/3.15.0 python/3.6.0

directory="sde-lorenz-96-scalability/sde-x-"
dynamical="lorenz-96"
num_xs=( 25 50 100 200 400 800 )

for num_x in "${num_xs[@]}"
do
    for ((i=1; i<=10; i++));
    do
        echo bsub -n 1 -W 2:00 -R "rusage[mem=2048]" -o ../data/${directory}${num_x}/${i}-out.log \
            python ./sde_main.py ../data/${directory}${num_x}/ ${dynamical} ${i}
        bsub -n 1 -W 2:00 -R "rusage[mem=2048]" -o ../data/${directory}${num_x}/${i}-out.log \
            python ./sde_main.py ../data/${directory}${num_x}/ ${dynamical} ${i}
        echo
    done
done


directory="sde-lorenz-96-scalability/ode-x-"

for num_x in "${num_xs[@]}"
do
    for ((i=1; i<=10; i++));
    do
        echo bsub -n 1 -W 2:00 -R "rusage[mem=2048]" -o ../data/${directory}${num_x}/${i}-out.log \
            python ./ode_main.py ../data/${directory}${num_x}/ ${dynamical} ${i}
        bsub -n 1 -W 2:00 -R "rusage[mem=2048]" -o ../data/${directory}${num_x}/${i}-out.log \
            python ./ode_main.py ../data/${directory}${num_x}/ ${dynamical} ${i}
        echo
    done
done
