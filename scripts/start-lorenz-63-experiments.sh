#!/usr/bin/env bash

module load new gcc/4.8.2 openblas/0.2.13_seq sqlite/3.15.0 python/3.6.0

directory="full"
dynamical="lorenz-63"

for ((repetition=1; repetition<=10; repetition++));
do
    for ((i=1; i<=100; i++));
    do
        echo bsub -n 1 -W 2:00 -R "rusage[mem=2048]" -o ../data/sde-lorenz-63/${directory}/${repetition}/${i}-out.log \
            python ./sde_main.py ../data/sde-lorenz-63/${directory}/${repetition}/ ${dynamical} ${i}
        bsub -n 1 -W 2:00 -R "rusage[mem=2048]" -o ../data/sde-lorenz-63/${directory}/${repetition}/${i}-out.log \
            python ./sde_main.py ../data/sde-lorenz-63/${directory}/${repetition}/ ${dynamical} ${i}
        echo
    done
done


directory="partial"

for ((repetition=1; repetition<=10; repetition++));
do
    for ((i=1; i<=100; i++));
    do
        echo bsub -n 1 -W 2:00 -R "rusage[mem=2048]" -o ../data/sde-lorenz-63/${directory}/${repetition}/${i}-out.log \
            python ./sde_main.py ../data/sde-lorenz-63/${directory}/${repetition}/ ${dynamical} ${i}
        bsub -n 1 -W 2:00 -R "rusage[mem=2048]" -o ../data/sde-lorenz-63/${directory}/${repetition}/${i}-out.log \
            python ./sde_main.py ../data/sde-lorenz-63/${directory}/${repetition}/ ${dynamical} ${i}
        echo
    done
done
