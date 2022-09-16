#!/bin/bash

# Specs.
SEED=2021
FOLD=0  # indicate testing fold (will be trained on the rest!)
RUNS=1  # number of runs (repetitions)
DATA=data/BRATST2
SAVE_FOLDER=results/train/fold${FOLD}

# Run.
mkdir -p ${SAVE_FOLDER}
for _ in $(seq 1 ${RUNS})
do
  python main_train.py \
  --data_root ${DATA} \
  --save_root ${SAVE_FOLDER} \
  --dataset BRATS \
  --n_sv 5000 \
  --fold ${FOLD} \
  --seed ${SEED}
done

