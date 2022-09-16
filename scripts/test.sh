#!/bin/bash

# Specs.
SEED=2021
FOLD=0  # testing on this fold
DATA=data/BRATST2
PRETRAINED=results/train/fold${FOLD}/model.pth
SAVE_FOLDER=results/test/fold${FOLD}
mkdir -p ${SAVE_FOLDER}

# Run.
python main_inference.py \
--data_root ${DATA} \
--save_root ${SAVE_FOLDER} \
--pretrained_root "${PRETRAINED}" \
--dataset BRATS \
--fold ${FOLD} \
--seed ${SEED}

# Note: EP2 is default, for EP1 set --EP1 True, --n_shot 3.

