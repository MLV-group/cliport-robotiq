#!/bin/bash

DATA_DIR=data_2fingered
DISP=False

echo "Generating dataset... Folder: $DATA_DIR"

# You can parallelize these depending on how much resources you have

############################
# Language-Conditioned Tasks

# LANG_TASKS='packing-boxes-pairs-seen-colors packing-boxes-pairs-unseen-colors packing-seen-google-objects-seq packing-unseen-google-objects-seq packing-seen-google-objects-group packing-unseen-google-objects-group'
# LANG_TASKS='packing-boxes-pairs-seen-colors packing-boxes-pairs-unseen-colors'

# for task in $LANG_TASKS
#     do
#         python cliport/demos.py n=1000 task=$task mode=train data_dir=$DATA_DIR disp=$DISP &
#         python cliport/demos.py n=100  task=$task mode=val   data_dir=$DATA_DIR disp=$DISP &
#         python cliport/demos.py n=100  task=$task mode=test  data_dir=$DATA_DIR disp=$DISP
#     done
# echo "Finished Language Tasks."


#########################
## Demo-Conditioned Tasks

# DEMO_TASKS='packing-boxes'
DEMO_TASKS='packing-boxes'

for task in $DEMO_TASKS
    do
        python cliport/demos.py n=1000 task=$task mode=train data_dir=$DATA_DIR disp=$DISP &
        python cliport/demos.py n=100  task=$task mode=val   data_dir=$DATA_DIR disp=$DISP &
        python cliport/demos.py n=100  task=$task mode=test  data_dir=$DATA_DIR disp=$DISP
    done
echo "Finished Demo Tasks."


