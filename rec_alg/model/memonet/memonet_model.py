# -*- coding:utf-8 -*-
import copy

import tensorflow as tf

from rec_alg.common import tf_utils
from rec_alg.common.utils import Utils
from rec_alg.components.inputs import build_input_features, input_from_feature_columns
from rec_alg.components.layers import DenseEmbeddingLayer, PredictionLayer, DNNLayer
from rec_alg.components.multi_hash_codebook_layer import MultiHashCodebookLayer
from rec_alg.components.multi_hash_codebook_kif_layer import MultiHashCodebookKIFLayer


class MemoNetModel(object):
    def __init__(self, feature_columns, params, embedding_size, embedding_l2_reg=0.0, embedding_dropout=0,
                 dnn_hidden_units=(), dnn_activation='relu', dnn_l2_reg=0.0, dnn_use_bn=False,
                 dnn_dropout=0.0, init_std=0.01, task='binary', seed=1024):
        super(MemoNetModel, self).__init__()
        tf.compat.v1.set_random_seed(seed=seed)

        self.feature_columns = feature_columns
        self.field_size = len(feature_columns)
        self.params = copy.deepcopy(params)

        self.embedding_size = embedding_size
        self.embedding_l2_reg = embedding_l2_reg
        self.embedding_dropout = embedding_dropout

        self.dnn_hidden_units = dnn_hidden_units
        self.dnn_activation = dnn_activation
        self.dnn_l2_reg = dnn_l2_reg
        self.dnn_use_bn = dnn_use_bn
        self.dnn_dropout = dnn_dropout

        self.init_std = init_std
        self.task = task
        self.seed = seed

        self.features = None
        self.inputs_list = None
        self.embeddings = None
        self.outputs = None

        self._init()
        return

    def _init(self):
        self.interact_mode = self.params.get("interact_mode", "fullhcnet")
        self.interaction_hash_embedding_buckets = self.params.get("interaction_hash_embedding_buckets", 100000)
        self.interaction_hash_embedding_size = self.params.get("interaction_hash_embedding_size", self.embedding_size)
        self.interaction_hash_embedding_bucket_mode = self.params.get("interaction_hash_embedding_bucket_mode",
                                                                      "hash-share")
        self.interaction_hash_embedding_num_hash = self.params.get("interaction_hash_embedding_num_hash", 2)
        self.interaction_hash_embedding_merge_mode = self.params.get("interaction_hash_embedding_merge_mode", "concat")
        self.interaction_hash_output_dims = self.params.get("interaction_hash_output_dims", 0)
        self.interaction_hash_embedding_float_precision = self.params.get(
            "interaction_hash_embedding_float_precision", 12)
        self.interaction_hash_embedding_interact_orders = self.params.get(
            "interaction_hash_embedding_interact_orders", (2,))
        self.interaction_hash_embedding_interact_modes = self.params.get(
            "interaction_hash_embedding_interact_modes", ("none",))
        self.interaction_hash_embedding_feature_metric = self.params.get("interaction_hash_embedding_feature_metric",
                                                                         "dimension")
        self.interaction_hash_embedding_feature_top_k = self.params.get("interaction_hash_embedding_feature_top_k", -1)
        return

    def get_model(self):
        self.features, self.inputs_list = self.get_inputs()
        self.embeddings = self.get_embeddings()

        interact_embeddings = [self.embeddings]
        if "fullhcnet" in self.interact_mode:
            multi_hash_codebook_layer = MultiHashCodebookLayer(
                name="multi_hash_codebook_layer",
                num_buckets=self.interaction_hash_embedding_buckets,
                embedding_size=self.interaction_hash_embedding_size,
                bucket_mode=self.interaction_hash_embedding_bucket_mode,
                init_std=self.init_std,
                l2_reg=self.embedding_l2_reg,
                seed=self.seed,
                num_hash=self.interaction_hash_embedding_num_hash,
                merge_mode=self.interaction_hash_embedding_merge_mode,
                output_dims=self.interaction_hash_output_dims,
                params=self.params,
                hash_float_precision=self.interaction_hash_embedding_float_precision,
                interact_orders=self.interaction_hash_embedding_interact_orders,
                interact_modes=self.interaction_hash_embedding_interact_modes,
            )

            top_inputs_list, top_embeddings = tf_utils.get_top_inputs_embeddings(
                feature_columns=self.feature_columns, features=self.features, embeddings=self.embeddings,
                feature_importance_metric=self.interaction_hash_embedding_feature_metric,
                feature_importance_top_k=self.interaction_hash_embedding_feature_top_k)

            interaction_hash_embeddings, interact_field_weights = multi_hash_codebook_layer(
                [top_inputs_list, top_embeddings])
            interact_embeddings.append(interaction_hash_embeddings)
        if "subsethcnet" in self.interact_mode:
            print("-----------------GetTopInputsAndEmbeddings------------------")
            top_inputs_list, top_embeddings, top_field_indexes = tf_utils.get_top_inputs_embeddings(
                feature_columns=self.feature_columns, features=self.features, embeddings=self.embeddings,
                feature_importance_metric=self.interaction_hash_embedding_feature_metric,
                feature_importance_top_k=self.interaction_hash_embedding_feature_top_k,
                return_feature_index=True,
            )

            multi_hash_codebook_kif_layer = MultiHashCodebookKIFLayer(
                name="multi_hash_codebook_kif_layer",
                field_size=self.field_size,
                top_field_indexes=top_field_indexes,
                num_buckets=self.interaction_hash_embedding_buckets,
                embedding_size=self.interaction_hash_embedding_size,
                bucket_mode=self.interaction_hash_embedding_bucket_mode,
                init_std=self.init_std,
                l2_reg=self.embedding_l2_reg,
                seed=self.seed,
                num_hash=self.interaction_hash_embedding_num_hash,
                merge_mode=self.interaction_hash_embedding_merge_mode,
                output_dims=self.interaction_hash_output_dims,
                params=self.params,
                hash_float_precision=self.interaction_hash_embedding_float_precision,
                interact_orders=self.interaction_hash_embedding_interact_orders,
                interact_modes=self.interaction_hash_embedding_interact_modes,
            )

            print("-----------------GetAllInputsAndEmbeddings------------------")
            all_inputs_list, all_embeddings, all_feature_indexes = tf_utils.get_top_inputs_embeddings(
                feature_columns=self.feature_columns, features=self.features, embeddings=self.embeddings,
                feature_importance_metric=self.interaction_hash_embedding_feature_metric,
                feature_importance_top_k=-1,
                return_feature_index=True,
            )

            interaction_hash_embeddings, interact_field_weights = multi_hash_codebook_kif_layer(
                [all_inputs_list, all_embeddings])
            interact_embeddings.append(interaction_hash_embeddings)

        self.outputs = self.to_predict(interact_embeddings)
        model = tf.keras.models.Model(inputs=self.inputs_list, outputs=self.outputs)
        return model

    def get_inputs(self):
        """
        inputs of keras model
        :return:
        """
        features = build_input_features(self.feature_columns)
        inputs_list = list(features.values())
        return features, inputs_list

    def get_embeddings(self, name_prefix=""):
        """
        sparse & dense feature embeddings
        :return:
        """
        init_std = self.init_std * (self.embedding_size ** -0.5)
        sparse_embedding_list, dense_value_list = input_from_feature_columns(self.features, self.feature_columns,
                                                                             self.embedding_size,
                                                                             self.embedding_l2_reg,
                                                                             init_std,
                                                                             self.seed,
                                                                             prefix=name_prefix, )
        sparse_embeddings = Utils.concat_func(sparse_embedding_list, axis=1)

        dense_embeddings = None
        if len(dense_value_list) > 0:
            dense_values = tf.stack(dense_value_list, axis=1)
            dense_embeddings = DenseEmbeddingLayer(
                embedding_size=self.embedding_size,
                init_std=init_std,
                embedding_l2_reg=self.embedding_l2_reg)(dense_values)

        if len(dense_value_list) > 0:
            embeddings = Utils.concat_func([sparse_embeddings, dense_embeddings], axis=1)
        else:
            embeddings = sparse_embeddings
        embeddings *= self.embedding_size ** 0.5
        
        # 3. dropout
        embeddings = tf.keras.layers.Dropout(self.embedding_dropout, name="origin_embeddings")(embeddings)
        return embeddings

    def to_predict(self, interact_embeddings):
        interact_embeddings = [tf.keras.layers.Flatten()(embeddings) for embeddings in interact_embeddings]
        dnn_inputs = Utils.concat_func(interact_embeddings, axis=1)
        dnn_outputs = DNNLayer(self.dnn_hidden_units, self.dnn_activation, self.dnn_l2_reg, self.dnn_dropout,
                               self.dnn_use_bn, self.seed)(dnn_inputs)
        final_logit = tf.keras.layers.Dense(1, use_bias=True, activation=None, )(dnn_outputs)
        predict_outputs = PredictionLayer(self.task, use_bias=False)(final_logit)
        return predict_outputs
