#!/bin/bash

# Set the path to SPlishSPlasHs DynamicBoundarySimulator in splishsplash_config.py
# before running this script

# output directories
OUTPUT_SCENES_DIR=scenes_200FPS
OUTPUT_DATA_DIR=data_200FPS

mkdir $OUTPUT_SCENES_DIR

# This script is purely sequential but it is recommended to parallelize the
# following loop, which generates the simulation data.
for seed in `seq 1 100`; do
        echo "$seed"
        python create_physics_scenes.py --output $OUTPUT_SCENES_DIR \
                                        --seed $seed \
                                        --default-viscosity \
                                        --default-density \
                                        --default-box \
#                                        --const-fluid-particles $CONST_FLUID_PARTICLES
done


# Transforms and compresses the data such that it can be used for training.
# This will also create the OUTPUT_DATA_DIR.
python create_physics_records.py --input $OUTPUT_SCENES_DIR \
                                 --output $OUTPUT_DATA_DIR \
                                 --splits 20


# Split data in train and validation set
mkdir $OUTPUT_DATA_DIR/train
mkdir $OUTPUT_DATA_DIR/valid

for seed in `seq -f "%04g" 1 50`; do
        mv $OUTPUT_DATA_DIR/sim_${seed}_*.msgpack.zst $OUTPUT_DATA_DIR/train
done

for seed in `seq -f "%04g" 51 100`; do
        mv $OUTPUT_DATA_DIR/sim_${seed}_*.msgpack.zst $OUTPUT_DATA_DIR/valid
done
