# End-to-end Recommendation Algorithm Package

The currently supported algorithms:

1. MemoNet: [MemoNet: MemoNet: Memorizing All Cross Features' Representations Efficiently via Multi-Hash Codebook Network for CTR Prediction](https://arxiv.org/abs/2211.01334)
2. FiBiNet++: [FiBiNet++: Reducing Model Size by Low Rank Feature Interaction Layer for CTR Prediction](https://arxiv.org/abs/2209.05016)
3. FiBiNet: [FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction](https://arxiv.org/abs/1905.09433) .

The following algorithms are planned to be supported:
1. GateNet: [GateNet: Gating-Enhanced Deep Network for Click-Through Rate Prediction](https://arxiv.org/abs/2007.03519)
2. ContextNet: [ContextNet: A Click-Through Rate Prediction Framework Using Contextual information to Refine Feature Embedding](https://arxiv.org/abs/2107.12025)
3. MaskNet: [MaskNet: Introducing Feature-Wise Multiplication to CTR Ranking Models by Instance-Guided Mask](https://arxiv.org/abs/2102.07619)

## Prerequisites

- Python >= 3.6.8
- TensorFlow-GPU == 1.14

## Getting Started

### Installation

- Install TensorFlow-GPU 1.14

- Clone this repo

### Dataset

- Links of datasets are:

  - https://www.kaggle.com/c/criteo-display-ad-challenge
  - https://www.kaggle.com/c/avazu-ctr-prediction
  - https://www.kaggle.com/c/kddcup2012-track2
  
- You can download the original datasets and preprocess them by yourself. Run `python -u -m rec_alg.preprocessing.{dataset_name}.{dataset_name}_process` to preprocess the datasets. `dataset_name` can be `criteo`, `avazu` or `kdd12`. 

- This repo also contains a demo dataset of criteo, which contains 100,000 samples and has been preprocessed. It is used to help demonstrate models here.

### Training

#### MemoNet
You can use `python -u -m rec_alg.model.memonet.run_memonet` to train a specific model on a dataset. Parameters could be found in the code.

#### FiBiNet & FiBiNet ++

You can use `python -u -m rec_alg.model.fibinet.run_fibinet --version {version} --config {config_path}` to train a specific model on a dataset. 

Some important parameters are list below, and other hyper-parameters can be found in the code. 

  - version: model version, supports `v1`, `++`, and `custom`, default to `++`. For `custom`, you can adjust all parameter values flexibly.
  - config_path: specifies the paths of the input/output files and the fields of the dataset. It is generated during dataset preprocessing. Support values: `./config/criteo/config_dense.json`, `./config/avazu/config_sparse.json`.
  - mode: running mode, supports `train`, `retrain`, `test`. 

## Acknowledgement

Part of the code comes from [DeepCTR](https://github.com/shenweichen/DeepCTR).

