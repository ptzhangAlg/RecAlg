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
config_path="./config/criteo/config_dense.json"
train_paths="part0,part1,part2,part3,part4,part5,part6,part7"
valid_paths="part8"
test_paths="part9"

# Embedding
embedding_size=10
embedding_l2_reg=0.0
embedding_dropout=0.0

# MLP
dnn_hidden_units="[400,400]"
dnn_activation="relu"
dnn_l2_reg=0.0
dnn_use_bn=False
dnn_dropout=0.0

# Interact mode
interact_mode="fullhcnet"

# HCNet
interaction_hash_embedding_buckets=1000000
interaction_hash_embedding_size=${embedding_size}
interaction_hash_embedding_bucket_mode="hash-share"
interaction_hash_embedding_num_hash=2
interaction_hash_embedding_merge_mode="concat"
interaction_hash_output_dims=0
interaction_hash_embedding_float_precision=12
interaction_hash_embedding_interact_orders="[2]"
interaction_hash_embedding_interact_modes="['senetsum']"
interaction_hash_embedding_feature_metric="dimension"
interaction_hash_embedding_feature_top_k=-1

# Train/Test setup
seed=1024
epochs=3
batch_size=1024
learning_rate=0.001
init_std=0.01
verbose=1
mode="train"
restore_epochs=[]
early_stopping=True
model_path="memonet_model/"${cmd}

# TODO
###########################################
# Different runs
###########################################
if [ $cmd == "memonet_criteo_hcnet-full-1e6-10-2-concat_output-10_orders-2-senetsum" ]; then
  interact_mode="fullhcnet"
  interaction_hash_embedding_buckets=1000000
  interaction_hash_embedding_size=10
  interaction_hash_embedding_num_hash=2
  interaction_hash_embedding_merge_mode="concat"
  interaction_hash_output_dims=10
  interaction_hash_embedding_interact_orders="[2]"
  interaction_hash_embedding_interact_modes="['senetsum']"
  epochs=1
elif [ $cmd == "memonet_criteo_hcnet-full-1e6-10-2-senetorigin_output-10_orders-2-senetsum" ]; then
  interact_mode="fullhcnet"
  interaction_hash_embedding_buckets=1000000
  interaction_hash_embedding_size=10
  interaction_hash_embedding_num_hash=2
  interaction_hash_embedding_merge_mode="senetorigin"
  interaction_hash_output_dims=10
  interaction_hash_embedding_interact_orders="[2]"
  interaction_hash_embedding_interact_modes="['senetsum']"
  epochs=1
elif [ $cmd == "memonet_avazu_hcnet-full-1e6-10-2-concat_output-10_orders-2-senetsum" ]; then
  config_path="./config/avazu/config_sparse.json"
  embedding_size=50
  interact_mode="fullhcnet"
  interaction_hash_embedding_buckets=1000000
  interaction_hash_embedding_size=50
  interaction_hash_embedding_num_hash=2
  interaction_hash_embedding_merge_mode="concat"
  interaction_hash_output_dims=50
  interaction_hash_embedding_interact_orders="[2]"
  interaction_hash_embedding_interact_modes="['senetsum']"
  learning_rate=0.0001
  epochs=1
elif [ $cmd == "memonet_avazu_hcnet-full-1e6-10-2-senetorigin_output-10_orders-2-senetsum" ]; then
  config_path="./config/avazu/config_sparse.json"
  embedding_size=50
  interact_mode="fullhcnet"
  interaction_hash_embedding_buckets=1000000
  interaction_hash_embedding_size=50
  interaction_hash_embedding_num_hash=2
  interaction_hash_embedding_merge_mode="senetorigin"
  interaction_hash_output_dims=50
  interaction_hash_embedding_interact_orders="[2]"
  interaction_hash_embedding_interact_modes="['senetsum']"
  learning_rate=0.0001
  epochs=1
elif [ $cmd == "memonet_kdd12_hcnet-full-5e5-10-2-concat_output-10_orders-2-senetsum" ]; then
  config_path="./config/kdd12/config_dense.json"
  embedding_size=10
  interact_mode="fullhcnet"
  interaction_hash_embedding_buckets=500000
  interaction_hash_embedding_size=10
  interaction_hash_embedding_num_hash=2
  interaction_hash_embedding_merge_mode="concat"
  interaction_hash_output_dims=10
  interaction_hash_embedding_interact_orders="[2]"
  interaction_hash_embedding_interact_modes="['senetsum']"
  learning_rate=0.001
  epochs=2
elif [ $cmd == "memonet_kdd12_hcnet-full-5e5-10-2-senetorigin_output-10_orders-2-senetsum" ]; then
  config_path="./config/kdd12/config_dense.json"
  embedding_size=10
  interact_mode="fullhcnet"
  interaction_hash_embedding_buckets=500000
  interaction_hash_embedding_size=10
  interaction_hash_embedding_num_hash=2
  interaction_hash_embedding_merge_mode="senetorigin"
  interaction_hash_output_dims=10
  interaction_hash_embedding_interact_orders="[2]"
  interaction_hash_embedding_interact_modes="['senetsum']"
  learning_rate=0.001
  epochs=2
else
  echo "****ERROR unknown cmd: $cmd..."
  exit 1
fi

if [ ! -d "./logs" ]; then
  mkdir logs
fi

python -u -m rec_alg.model.memonet.run_memonet \
  --config_path ${config_path} \
  --train_paths ${train_paths} \
  --valid_paths ${valid_paths} \
  --test_paths ${test_paths} \
  --embedding_size ${embedding_size} \
  --embedding_l2_reg ${embedding_l2_reg} \
  --embedding_dropout ${embedding_dropout} \
  --dnn_hidden_units ${dnn_hidden_units} \
  --dnn_activation ${dnn_activation} \
  --dnn_l2_reg ${dnn_l2_reg} \
  --dnn_use_bn ${dnn_use_bn} \
  --dnn_dropout ${dnn_dropout} \
  --interact_mode ${interact_mode} \
  --interaction_hash_embedding_buckets ${interaction_hash_embedding_buckets} \
  --interaction_hash_embedding_size ${interaction_hash_embedding_size} \
  --interaction_hash_embedding_bucket_mode ${interaction_hash_embedding_bucket_mode} \
  --interaction_hash_embedding_num_hash ${interaction_hash_embedding_num_hash} \
  --interaction_hash_embedding_merge_mode ${interaction_hash_embedding_merge_mode} \
  --interaction_hash_output_dims ${interaction_hash_output_dims} \
  --interaction_hash_embedding_float_precision ${interaction_hash_embedding_float_precision} \
  --interaction_hash_embedding_interact_orders ${interaction_hash_embedding_interact_orders} \
  --interaction_hash_embedding_interact_modes ${interaction_hash_embedding_interact_modes} \
  --interaction_hash_embedding_feature_metric ${interaction_hash_embedding_feature_metric} \
  --interaction_hash_embedding_feature_top_k ${interaction_hash_embedding_feature_top_k} \
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
  >>./logs/memonet_${dt}_${cmd}.log 2>&1

echo "running ..."
