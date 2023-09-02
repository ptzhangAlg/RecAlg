#!/usr/bin/env python
# -*- coding:utf-8 -*-
import copy
import itertools

import tensorflow as tf
from tensorflow.python.keras.layers import Layer, Embedding
from tensorflow.python.keras.regularizers import l2

from rec_alg.common.utils import Utils
from rec_alg.components.inputs import get_embedding_initializer
from rec_alg.components.layers import StrongHash, SENETLayer


class MultiHashCodebookLayer(Layer):
    def __init__(self, num_buckets, embedding_size, bucket_mode="hash-share", initializer_mode="random_normal",
                 init_std=0.01, l2_reg=0.0, seed=1024, num_hash=1, merge_mode="concat", output_dims=0, params={},
                 hash_float_precision=12, interact_orders=(2,), interact_modes=("senetsum",), **kwargs):
        """
        Implement of multi-Hash Codebook Network (HCNet) in MemoNet, only supports input of all features
        :param num_buckets: num of codewords
        :param embedding_size: dimension of codeword
        :param bucket_mode: mode of codeword, support hash-share, hash-private
        :param initializer_mode: initializer of codebook
        :param init_std: init std of codebook
        :param l2_reg: l2 reg of code book
        :param seed: seed
        :param num_hash: num of hash functions
        :param merge_mode: merge mode of different codewords of a feature, support concat, senetorigin
        :param output_dims: output dim of HCNet
        :param params: expand params
        :param hash_float_precision: precision for float inputs
        :param interact_orders: orders of interaction. For example [2,3] means 2-order and 3-order interactions
        :param interact_modes: mode of interaction, supports "sum", "senetsum"
        :param kwargs:
        """
        self.num_buckets = num_buckets
        self.embedding_size = embedding_size
        self.bucket_mode = bucket_mode
        self.initializer_mode = initializer_mode
        self.init_std = init_std
        self.l2_reg = l2_reg
        self.seed = seed
        self.num_hash = num_hash
        self.merge_mode = merge_mode
        self.params = copy.deepcopy(params)
        self.hash_float_precision = hash_float_precision
        self.interact_orders = interact_orders
        self.interact_modes = list(interact_modes) + [interact_modes[-1] for _ in
                                                      range(len(interact_modes), len(interact_orders))]
        self.output_dims = self._get_output_dims(output_dims=output_dims)
        
        self.interact_indexes = None  # [order_n=[field_x=[target_idx]]]
        self.senet_layer = None
        self.field_interaction_idx = None
        # Max 11 Hash Functions
        self.hash_keys = [[7744, 1822],
                          [423, 6649],
                          [3588, 8319],
                          [8220, 7283],
                          [1965, 9209],
                          [4472, 1935],
                          [3987, 4403],
                          [2379, 2870],
                          [5638, 2954],
                          [2211, 2],
                          [6075, 9105]]
        self.field_size = None
        self.hash_layers = []
        self.embedding_layer = None
        self.hash_merge_layer = None
        self.interact_mode_layers = []
        self.transform_layer = None
        self.field_tokens = []
        self._init()
        super(MultiHashCodebookLayer, self).__init__(**kwargs)
    
    def _init(self):
        self.outer_interact_mode = self.params.get("interact_mode", None)
        return
    
    def build(self, input_shape):
        self.field_size = len(input_shape[0])
        self.interact_indexes = [self.get_field_interaction_idx(order_n, self.field_size)
                                 for order_n in self.interact_orders]
        
        # Hash Layers
        strong_hash = True if self.num_hash > 1 else False
        for i in range(self.num_hash):
            hash_layer = StrongHash(num_buckets=self.num_buckets, mask_zero=False, strong=strong_hash,
                                    key=self.hash_keys[i])
            self.hash_layers.append(hash_layer)
        
        # Codebooks
        self.embedding_layer = []
        num_embeddings = 1
        if self.bucket_mode == "hash-share":
            num_embeddings = 1
        elif self.bucket_mode == "hash-private":
            num_embeddings = self.num_hash
        for _ in self.interact_orders:
            self.embedding_layer.append([Embedding(input_dim=self.num_buckets, output_dim=self.embedding_size,
                                                   embeddings_initializer=get_embedding_initializer(
                                                       initializer_mode=self.initializer_mode,
                                                       mean=0.0,
                                                       stddev=self.init_std,
                                                       seed=self.seed),
                                                   embeddings_regularizer=l2(self.l2_reg))
                                         for _ in range(num_embeddings)])
        
        # Linear Memory Restoring (LMR) and Attentive Memory Restoring (AMR)
        if "senetorigin" in self.merge_mode:
            self.senet_layer = [SENETLayer(
                senet_squeeze_mode="bit", senet_reduction_ratio=1.0, senet_excitation_mode="bit",
                senet_activation="none", seed=self.seed,
                output_weights=True, output_field_size=self.num_hash, output_embedding_size=self.embedding_size
            ) for _ in self.interact_orders]
        self.transform_layer = tf.keras.layers.Dense(self.output_dims, activation=None,
                                                     use_bias=False, name="hash_merge_final_transform")
        
        # Feature Shrinking-Global Attentive Shrinking(GAS)
        for idx, interact_mode in enumerate(self.interact_modes):
            num_interact = len(list(itertools.combinations(range(self.field_size), self.interact_orders[idx])))
            interact_mode_layer = SENETLayer(
                senet_squeeze_mode="bit", senet_reduction_ratio=1.0, senet_excitation_mode="vector",
                senet_activation="none", seed=self.seed,
                output_weights=True, output_field_size=num_interact)
            self.interact_mode_layers.append(interact_mode_layer)
        
        for i in range(self.field_size):
            self.field_tokens.append(tf.constant(str(i), dtype=tf.string, shape=(1, 1)))
        super(MultiHashCodebookLayer, self).build(input_shape)  # Be sure to call this somewhere!
    
    def call(self, inputs, training=None, **kwargs):
        """
        :param placeholder_inputs: (?, length)
        :param origin_embeddings
        :param training:
        :param kwargs:
        :return:
        """
        placeholder_inputs, origin_embeddings = inputs[0], inputs[1]
        input_list = []
        batch_size = tf.shape(placeholder_inputs[0])[0]
        # 1. Multi-Hash Addressing
        # 1.1. Obtain all cross features
        for i in range(self.field_size):
            field_token = tf.tile(self.field_tokens[i], [batch_size, 1])
            if placeholder_inputs[i].dtype == tf.float32 or placeholder_inputs[i].dtype == tf.float16 or \
                    placeholder_inputs[i].dtype == tf.float64:
                item = tf.strings.as_string(placeholder_inputs[i], precision=self.hash_float_precision)
            else:
                item = tf.strings.as_string(placeholder_inputs[i], )
            # Process VarLenSparse Feature
            if item.shape[-1].value > 1:
                item = tf.expand_dims(tf.strings.reduce_join(item, axis=-1, separator="-"), axis=1)
            field_item = tf.strings.reduce_join([field_token, item], axis=0, separator="_")
            input_list.append(field_item)
        
        interact_tokens = []
        for order_n in self.interact_orders:
            tokens = self.get_high_order_tokens(input_list=input_list, order_n=order_n, field_size=self.field_size)
            interact_tokens.append(tokens)
        
        interact_embeddings = []
        interact_field_weights = []
        for idx, tokens in enumerate(interact_tokens):
            # 2. Multi-Hash Address && Memory Restoring
            embeddings = self.get_embeddings_from_tokens(tokens, origin_embeddings, interact_order_idx=idx)
            # 3. Feature Shrinking
            if self.interact_modes[idx] == "sum":
                # [batch, fields, field_interact_num, embedding_size]
                field_embeddings = tf.gather(embeddings, self.interact_indexes[idx], axis=1)
                # [batch,fields, embedding_size]
                embeddings = tf.reduce_sum(field_embeddings, axis=-2, keepdims=False)
            elif "senetsum" in self.interact_modes[idx]:
                # [batch, fields, field_interact_num, embedding_size]
                field_embeddings = tf.gather(embeddings, self.interact_indexes[idx], axis=1)
                # [batch, num_interact, vector=1|bit=output_dims]
                weights = self.interact_mode_layers[idx](origin_embeddings)
                # [batch, fields, field_interact_num, 1]
                field_weights = tf.gather(weights, self.interact_indexes[idx], axis=1)
                if "softmax" in self.interact_modes[idx]:
                    temperature = self.get_float_from_param(self.interact_modes[idx], default_value=1.0)
                    field_weights = tf.nn.softmax(field_weights / temperature, axis=-2)
                interact_field_weights.append(field_weights)
                # [batch, fields, embedding_size]
                weighted_field_embeddings = field_embeddings * field_weights
                # [batch,fields, embedding_size]
                embeddings = tf.reduce_sum(weighted_field_embeddings, axis=-2, keepdims=False)
            interact_embeddings.append(embeddings)
        
        outputs = tf.concat(interact_embeddings, axis=1)
        return outputs, interact_field_weights
    
    def get_float_from_param(self, param, default_value=0.0):
        value = default_value
        try:
            value = float(param.split('-')[-1])
        except:
            pass
        return value
    
    @staticmethod
    def get_high_order_tokens(input_list, order_n, field_size):
        """
        Get high order tokens from input_list
        :param input_list:
        :param order_n:
        :param field_size:
        :return:
        """
        interact_token_list = []
        for idx_tuples in itertools.combinations(range(field_size), order_n):
            input_items = [input_list[idx] for idx in idx_tuples]
            interact_token = tf.strings.reduce_join(input_items, axis=0, separator="_")
            interact_token_list.append(interact_token)
        tokens = Utils.concat_func(interact_token_list, axis=1)
        return tokens
    
    def get_embeddings_from_tokens(self, tokens, origin_embeddings, interact_order_idx=0):
        # Multi-Hash Addressing.
        hash_embedding_list = []
        for i in range(self.num_hash):
            hash_idx = self.hash_layers[i](tokens)
            if self.bucket_mode == "hash-share":
                hash_embedding = self.embedding_layer[interact_order_idx][0](hash_idx)
            elif self.bucket_mode == "hash-private":
                hash_embedding = self.embedding_layer[interact_order_idx][i](hash_idx)
            else:
                raise Exception("Unknown bucket_mode: {}".format(self.bucket_mode))
            hash_embedding_list.append(hash_embedding)  # [(batch, num_interact, embeddings)]
        
        # Memory Restoring
        if "concat" in self.merge_mode:
            embeddings = Utils.concat_func(hash_embedding_list, axis=-1)  # [batch, num_interact, num_hash*embeddings]
        elif "senetorigin" in self.merge_mode:
            reweight_embeddings = tf.stack(hash_embedding_list, axis=-2)  # [batch, num_interacts, num_hash, embeddings]
            
            embeddings = self._merge_by_senet_origin(input_embeddings=origin_embeddings,  # [batch, fields, embeddings]
                                                     reweight_embeddings=reweight_embeddings,
                                                     interact_order_idx=interact_order_idx)
        else:
            raise Exception("MultiHashCodebookLayer: unknown hash_merge_mode: {}".format(self.merge_mode))
        
        # Used to decrease outputs dimensions
        embeddings = self.transform_layer(embeddings)
        return embeddings
    
    def _merge_by_senet_origin(self, input_embeddings, reweight_embeddings, interact_order_idx=0):
        num_interacts = reweight_embeddings.shape[1].value
        # [batch*num_interact, num_hash, embeddings]
        reweight_embeddings = tf.reshape(reweight_embeddings, (-1, self.num_hash, self.embedding_size))
        origin_embedding_size = input_embeddings.shape[-1].value
        split_embeddings = tf.split(input_embeddings, self.field_size, axis=1)  # [(batch, 1, embeddings)]
        interact_embedding_list = []
        for idx_tuples in itertools.combinations(range(self.field_size), self.interact_orders[interact_order_idx]):
            # [batch, order_n, embeddings]
            interact_input_embeddings = [split_embeddings[idx] for idx in idx_tuples]
            interact_embedding = Utils.concat_func(interact_input_embeddings, axis=1)
            interact_embedding_list.append(interact_embedding)
        interact_embeddings = tf.stack(interact_embedding_list, axis=1)  # [batch, num_interact, order_n, embeddings]
        
        # Inputs of weights net, [batch*num_interacts, order_n, embeddings]
        inputs = tf.reshape(interact_embeddings, (-1, interact_embeddings.shape[-2], origin_embedding_size))
        # Weights
        weights = self.senet_layer[interact_order_idx](inputs)  # [batch*num_interact, fields, fields=1|bit=embeddings]
        # Reweight
        reweight_outputs = reweight_embeddings * weights
        # Reshape
        reshape_outputs = tf.reshape(reweight_outputs, (-1, num_interacts, self.num_hash, self.embedding_size))
        # Concat [batch, num_interact, num_hash*embeddings]
        outputs = tf.reshape(reshape_outputs, (-1, num_interacts, self.num_hash * self.embedding_size))
        return outputs
    
    def _get_output_dims(self, output_dims):
        output_dims = output_dims if output_dims and output_dims > 0 else self.embedding_size
        return output_dims
    
    @staticmethod
    def get_field_interaction_idx(order_n, field_size):
        # 2-order: [(1,2), (1,3), ..., (1, N), ..., (N-1,N)]
        field_interaction_idx = [[] for _ in range(field_size)]
        
        idx_dict = dict()
        target_idx = 0
        for idx_tuples in itertools.combinations(range(field_size), order_n):
            for idx in idx_tuples:
                field_interaction_idx[idx].append(target_idx)
            idx_dict[target_idx] = idx_tuples
            target_idx += 1
        
        # print(idx_dict)
        for i in range(field_size):
            field_interaction_idx[i] = sorted(field_interaction_idx[i])
        return field_interaction_idx
    
    def compute_output_shape(self, input_shape):
        output_dims = self.field_size * (self.field_size - 1) // 2
        if "concat" in self.merge_mode:
            shape = (input_shape[0][0], int(output_dims * self.num_hash), self.embedding_size)
        else:
            shape = (input_shape[0][0], output_dims, self.embedding_size)
        return shape
    
    def get_config(self, ):
        config = {'num_buckets': self.num_buckets, 'embedding_size': self.embedding_size,
                  'initializer_mode': self.initializer_mode, 'init_std': self.init_std, 'l2_reg': self.l2_reg,
                  'seed': self.seed, "num_hash": self.num_hash, "hash_merge_mode": self.merge_mode}
        base_config = super(MultiHashCodebookLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
