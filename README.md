# CLIPort 기반의 2-fingered gripper 물체 조작 시뮬레이터

## 베이스라인 모델:

[**CLIPort: What and Where Pathways for Robotic Manipulation**](https://arxiv.org/pdf/2109.12098.pdf)  
[Mohit Shridhar](https://mohitshridhar.com/), [Lucas Manuelli](http://lucasmanuelli.com/), [Dieter Fox](https://homes.cs.washington.edu/~fox/)  
[CoRL 2021](https://www.robot-learning.org/)

## 환경 설정

가상 환경 설정 및 파이썬 패키지 설치:

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

### Google 물체 다운로드

```bash
sh scripts/google_objects_download.sh
```

Credit: [Google](#acknowledgements).

## 학습과 평가

1. `demos.py` 파일로 `train`, `val`, `test` 데이터셋 생성
2. `train.py`로 모델 학습
3. `eval.py`을 실행하여 'val' 데이터셋에 대해 가장 좋은 가중치를 획득 (`*val-results.json`에 저장됨)
4. 'test' 데이터셋에 대해 `eval.py`을 실행하여 평가

### 데이터셋 생성

#### 단일 작업

1000개의 `train` 데이터셋을 `stack-block-pyramid-seq-seen-colors` 작업에 대해 만든다고 할 때, `$CLIPORT_ROOT/data_2fingered`에 저장됨:

```bash
python cliport/demos.py n=1000 \
                        task=stack-block-pyramid-seq-seen-colors \
                        mode=train
```

#### 전체 데이터셋

```bash
sh scripts/generate_dataset_2fingered.sh data
```

### 단일 작업 학습 및 평가

#### 학습

`stack-block-pyramid-seq-seen-colors` 작업에 대해 1000개의 데이터셋을 200k iteration으로 학습:

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

#### 검증

validation 데이터셋에 대해 모든 가중치를 평가하고 `exps/<task>-train/checkpoints/<task>-val-results.json`에 결과 저장:

```bash
python cliport/eval.py eval_task=stack-block-pyramid-seq-seen-colors \
                       agent=cliport \
                       mode=val \
                       n_demos=100 \
                       train_demos=1000 \
                       checkpoint_type=val_missing \
                       exp_folder=exps
```

#### 평가

test 데이터셋에 대해 가장 좋은 성능을 보인 가중치를 평가하고 `exps/<task>-train/checkpoints/<task>-test-results.json`에 결과 저장:

```bash
python cliport/eval.py eval_task=stack-block-pyramid-seq-seen-colors \
                       agent=cliport \
                       mode=test \
                       n_demos=100 \
                       train_demos=1000 \
                       checkpoint_type=test_best \
                       exp_folder=exps
```

### 여러 개의 작업 학습 및 평가

#### 학습

`task=multi-language-conditioned`, `task=multi-attr-packing-box-pairs-unseen-colors` 등을 설정

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

#### 검증

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

#### 평가

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

## 비디오 저장

`record.save_video=True`를 설정하여 비디오 저장 가능 (시간 오래 걸림):

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

`${model_dir}/${exp_folder}/${eval_task}-${agent}-n${train_demos}-train/videos/`에 저장
# cliport-robotiq
