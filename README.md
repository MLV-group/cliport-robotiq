# CLIPort-based 2-fingered gripper object manipulation simulator

## Baseline:

[**CLIPort: What and Where Pathways for Robotic Manipulation**](https://arxiv.org/pdf/2109.12098.pdf)  
[Mohit Shridhar](https://mohitshridhar.com/), [Lucas Manuelli](http://lucasmanuelli.com/), [Dieter Fox](https://homes.cs.washington.edu/~fox/)  
[CoRL 2021](https://www.robot-learning.org/)

## Environment setting

virtual environment settings and installing python pakages:

```bash
# setup virtualenv with whichever package manager you prefer
virtualenv -p $(which python3.8) --system-site-packages cliport_env
source cliport_env/bin/activate
pip install --upgrade pip

cd cliport
pip install -r requirements.txt

export CLIPORT_ROOT=$(pwd)
python setup.py develop
```

### Google object download

```bash
sh scripts/google_objects_download.sh
```

Credit: [Google](#acknowledgements).

## Training and Evaluation

1. Generate `train`, `val`, `test` dataset by 'demos.py'
2. Train by `train.py`
3. `eval.py` to obtain best checkpoint via 'val' dataset (save in `*val-results.json`)
4. Evaluate on 'test' dataset by running `eval.py`

### Dataset generation

#### single task

Train model on 1000 demos of `train` dataset with `stack-block-pyramid-seq-seen-colors` task (saved in `$CLIPORT_ROOT/data_2fingered`):

```bash
python cliport/demos.py n=1000 \
                        task=stack-block-pyramid-seq-seen-colors \
                        mode=train
```

#### Generate whole tasks

```bash
sh scripts/generate_dataset_2fingered.sh data
```

### Single task training and evaluation

#### Train

Train `stack-block-pyramid-seq-seen-colors` task on 1000 demos with 200k iteration:

```bash
python cliport/train.py train.task=stack-block-pyramid-seq-seen-colors \
                        train.agent=cliport \
                        train.attn_stream_fusion_type=add \
                        train.trans_stream_fusion_type=conv \
                        train.lang_fusion_type=mult \
                        train.n_demos=1000 \
                        train.n_steps=201000 \
                        train.exp_folder=exps \
                        dataset.cache=False
```

#### Validation

Evaluate on 'val' dataset by saving the result in `exps/<task>-train/checkpoints/<task>-val-results.json`:

```bash
python cliport/eval.py eval_task=stack-block-pyramid-seq-seen-colors \
                       agent=cliport \
                       mode=val \
                       n_demos=100 \
                       train_demos=1000 \
                       checkpoint_type=val_missing \
                       exp_folder=exps
```

#### Evaluation

Evaluate on 'test' dataset with best checkpoint (results saved in `exps/<task>-train/checkpoints/<task>-test-results.json`):

```bash
python cliport/eval.py eval_task=stack-block-pyramid-seq-seen-colors \
                       agent=cliport \
                       mode=test \
                       n_demos=100 \
                       train_demos=1000 \
                       checkpoint_type=test_best \
                       exp_folder=exps
```

### Multi-task training and evaluation

#### Train

Set `task=multi-language-conditioned`, `task=multi-attr-packing-box-pairs-unseen-colors`

```bash
python cliport/train.py train.task=multi-language-conditioned \
                        train.agent=cliport \
                        train.attn_stream_fusion_type=add \
                        train.trans_stream_fusion_type=conv \
                        train.lang_fusion_type=mult \
                        train.n_demos=1000 \
                        train.n_steps=601000 \
                        dataset.cache=False \
                        train.exp_folder=exps \
                        dataset.type=multi
```

#### Validation

```bash
python cliport/eval.py model_task=multi-language-conditioned \
                       eval_task=stack-block-pyramid-seq-seen-colors \
                       agent=cliport \
                       mode=val \
                       n_demos=100 \
                       train_demos=1000 \
                       checkpoint_type=val_missing \
                       type=single \
                       exp_folder=exps
```

#### Evaluation

```bash
python cliport/eval.py model_task=multi-language-conditioned \
                       eval_task=stack-block-pyramid-seq-seen-colors \
                       agent=cliport \
                       mode=test \
                       n_demos=100 \
                       train_demos=1000 \
                       checkpoint_type=test_best \
                       type=single \
                       exp_folder=exps
```

## Saving video

By setting `record.save_video=True`: (This takes a lot of time)

```bash
python cliport/eval.py model_task=multi-language-conditioned \
                       eval_task=stack-block-pyramid-seq-seen-colors \
                       agent=cliport \
                       mode=test \
                       n_demos=10 \
                       train_demos=100 \
                       exp_folder=cliport_exps \
                       checkpoint_type=test_best \
                       update_results=True \
                       disp=True \
                       record.save_video=True
```

Saved in `${model_dir}/${exp_folder}/${eval_task}-${agent}-n${train_demos}-train/videos/`
