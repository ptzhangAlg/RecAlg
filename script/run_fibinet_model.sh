#!/bin/bash
set -x

#export CUDA_VISIBLE_DEVICES=1
export TF_FORCE_GPU_ALLOW_GROWTH=true

dt=$(date "+%Y%m%d")

cmd=$1
echo "*****cmd:$cmd"

###########################################
# Parameters
###########################################
version="++"
config_path="./config/criteo/config_dense.json"
train_paths="part0,part1,part2,part3,part4,part5,part6,part7"
valid_paths="part8"
test_paths="part9"

# Embedding
embedding_size=10
embedding_l2_reg=0.0
embedding_dropout=0.0
sparse_embedding_norm_type="bn"
dense_embedding_norm_type="layer_norm"
dense_embedding_share_params=False

# SENet
senet_squeeze_mode="group_mean_max"
senet_squeeze_group_num=2
senet_squeeze_topk=1
senet_reduction_ratio=3.0
senet_excitation_mode="bit"
senet_activation="none"
senet_use_skip_connection=True
senet_reweight_norm_type="ln"

# Bilinear-Interaction
origin_bilinear_type="all_ip"
origin_bilinear_dnn_units="[50]"
origin_bilinear_dnn_activation="linear"
senet_bilinear_type="none"

# MLP
dnn_hidden_units="[400,400,400]"
dnn_activation="relu"
dnn_l2_reg=0.0
dnn_use_bn=False
dnn_dropout=0.0

# Linear Part
enable_linear=False
linear_l2_reg=0.0

# Train/Test setup
seed=1024
epochs=3
batch_size=1024
learning_rate=0.0001
init_std=0.01
verbose=1
mode="train"
restore_epochs=[]
early_stopping=False
model_path="fibinet_model/"${cmd}

###########################################
# Different runs
###########################################
if [ $cmd == "fibinet_v1_criteo" ]; then
  version="v1"
elif [ $cmd == "fibinet++_criteo" ]; then
  version="++"
elif [ $cmd == "fibinet_v1_avazu" ]; then
  embedding_size=50
  learning_rate=0.001
  config_path="./config/avazu/config_sparse.json"
  version="v1"
elif [ $cmd == "fibinet++_avazu" ]; then
  embedding_size=50
  learning_rate=0.001
  config_path="./config/avazu/config_sparse.json"
  version="++"
else
  echo "****ERROR unknown cmd: $cmd..."
  exit 1
fi

if [ ! -d "./logs" ]; then
  mkdir logs
fi

python -u -m rec_alg.model.fibinet.run_fibinet \
  --version ${version} \
  --config_path ${config_path} \
  --train_paths ${train_paths} \
  --valid_paths ${valid_paths} \
  --test_paths ${test_paths} \
  --embedding_size ${embedding_size} \
  --embedding_l2_reg ${embedding_l2_reg} \
  --embedding_dropout ${embedding_dropout} \
  --sparse_embedding_norm_type ${sparse_embedding_norm_type} \
  --dense_embedding_norm_type ${dense_embedding_norm_type} \
  --dense_embedding_share_params ${dense_embedding_share_params} \
  --senet_squeeze_mode ${senet_squeeze_mode} \
  --senet_squeeze_group_num ${senet_squeeze_group_num} \
  --senet_squeeze_topk ${senet_squeeze_topk} \
  --senet_reduction_ratio ${senet_reduction_ratio} \
  --senet_excitation_mode ${senet_excitation_mode} \
  --senet_activation ${senet_activation} \
  --senet_use_skip_connection ${senet_use_skip_connection} \
  --senet_reweight_norm_type ${senet_reweight_norm_type} \
  --origin_bilinear_type ${origin_bilinear_type} \
  --origin_bilinear_dnn_units ${origin_bilinear_dnn_units} \
  --origin_bilinear_dnn_activation ${origin_bilinear_dnn_activation} \
  --senet_bilinear_type ${senet_bilinear_type} \
  --dnn_hidden_units ${dnn_hidden_units} \
  --dnn_activation ${dnn_activation} \
  --dnn_l2_reg ${dnn_l2_reg} \
  --dnn_use_bn ${dnn_use_bn} \
  --dnn_dropout ${dnn_dropout} \
  --enable_linear ${enable_linear} \
  --linear_l2_reg ${linear_l2_reg} \
  --seed ${seed} \
  --epochs ${epochs} \
  --batch_size ${batch_size} \
  --learning_rate ${learning_rate} \
  --init_std ${init_std} \
  --verbose ${verbose} \
  --mode ${mode} \
  --restore_epochs ${restore_epochs} \
  --early_stopping ${early_stopping} \
  --model_path ${model_path} \
  >>./logs/fibinet_${dt}_${cmd}.log 2>&1

echo "running ..."
