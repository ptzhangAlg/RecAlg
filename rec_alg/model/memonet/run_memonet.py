import argparse
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow import keras

from rec_alg.common.utils import Utils
from rec_alg.model.base_model import BaseModel
from rec_alg.model.memonet.memonet_model import MemoNetModel


# tf.enable_eager_execution()


class MemoNetRunner(BaseModel):
    
    def __init__(self):
        # Get args
        self.args = self.parse_args()
        print("Args: ", self.args.__dict__)
        args_dict = self.args.__dict__.copy()
        config_path = args_dict.pop("config_path")
        super(MemoNetRunner, self).__init__(config_path=config_path, **args_dict)
        return
    
    def parse_args(self):
        parser = argparse.ArgumentParser()
        # 1. Run setup
        parser.add_argument('--config_path', type=str, default="./config/criteo/config_dense.json",
                            help="#config path: path of config file which includes info of dataset features")
        parser.add_argument('--train_paths', type=Utils.str2liststr,
                            default="part0,part1,part2,part3,part4,part5,part6,part7",
                            help='#train_paths: training directories split with comma')
        parser.add_argument('--valid_paths', type=Utils.str2liststr, default="part8",
                            help='#valid_paths: validation directories split with comma')
        parser.add_argument('--test_paths', type=Utils.str2liststr, default="part9",
                            help='#test_paths: testing directories split with comma')
        
        # 2. Model architecture
        # 2.1. Embedding
        parser.add_argument('--embedding_size', type=int, default=10,
                            help='#embedding_size: feature embedding size')
        parser.add_argument('--embedding_l2_reg', type=float, default=0.0,
                            help='#embedding_l2_reg: L2 regularizer strength applied to embedding')
        parser.add_argument('--embedding_dropout', type=float, default=0.0,
                            help='#embedding_dropout: the probability of dropping out on embedding')
        
        # 2.4. DNN part
        parser.add_argument('--dnn_hidden_units', type=Utils.str2list, default=[400, 400],
                            help='#dnn_hidden_units: layer number and units in each layer of DNN')
        parser.add_argument('--dnn_activation', type=str, default='relu',
                            help='#dnn_activation: activation function used in DNN')
        parser.add_argument('--dnn_l2_reg', type=float, default=0.0,
                            help='#dnn_l2_reg: L2 regularizer strength applied to DNN')
        parser.add_argument('--dnn_use_bn', type=Utils.str2bool, default=False,
                            help='#dnn_use_bn: whether to use BatchNormalization before activation or not in DNN')
        parser.add_argument('--dnn_dropout', type=float, default=0.0,
                            help='#dnn_dropout: the probability of dropping out on each layer of DNN')
        
        # 2.3. Interact-mode
        parser.add_argument('--interact_mode', type=str, default='fullhcnet',
                            help='#interact_mode: str, fullhcnet, subsethcnet')
        
        # 2.4. experience-embedding-hash
        parser.add_argument('--interaction_hash_embedding_buckets', type=int, default=100000,
                            help='#interaction_hash_embedding_buckets: int')
        parser.add_argument('--interaction_hash_embedding_size', type=int, default=10,
                            help='#interaction_hash_embedding_size: int')
        parser.add_argument('--interaction_hash_embedding_bucket_mode', type=str, default="hash-share",
                            help='#interaction_hash_embedding_bucket_mode: str')
        parser.add_argument('--interaction_hash_embedding_num_hash', type=int, default=2,
                            help='#interaction_hash_embedding_num_hash: int')
        parser.add_argument('--interaction_hash_embedding_merge_mode', type=str, default="concat",
                            help='#interaction_hash_embedding_merge_mode:str')
        parser.add_argument('--interaction_hash_output_dims', type=int, default=0,
                            help='#interaction_hash_output_dims: int')
        parser.add_argument('--interaction_hash_embedding_float_precision', type=int, default=12)
        parser.add_argument('--interaction_hash_embedding_interact_orders', type=Utils.str_to_type, default=[2, ],
                            help='#interaction_hash_embedding_interact_orders: list')
        parser.add_argument('--interaction_hash_embedding_interact_modes', type=Utils.str_to_type,
                            default=["senetsum", ], help='#interaction_hash_embedding_interact_modes: list')
        parser.add_argument('--interaction_hash_embedding_feature_metric', type=str, default="dimension")
        parser.add_argument('--interaction_hash_embedding_feature_top_k', type=int, default=-1)
        
        # 3. Train/Valid/Test setup
        parser.add_argument('--seed', type=int, default=1024, help='#seed: integer ,to use as random seed.')
        parser.add_argument('--epochs', type=int, default=3)
        parser.add_argument('--batch_size', type=int, default=1024)
        parser.add_argument('--learning_rate', type=float, default=0.001)
        parser.add_argument('--init_std', type=float, default=0.01)
        parser.add_argument('--verbose', type=int, default=1)
        parser.add_argument("--mode", type=str, default="train", help="support: train, retrain, test")
        parser.add_argument('--restore_epochs', type=Utils.str2list, default=[],
                            help="restore weights from checkpoint, format like np.arange(), eg. [1, 5, 1]")
        parser.add_argument("--early_stopping", type=Utils.str2bool, default=True, help="enable early stopping")
        parser.add_argument("--model_path", type=str, default="rec_alg", help="model_path, to avoid being covered")
        return parser.parse_args()
    
    def create_model(self):
        memonet_model = MemoNetModel(feature_columns=self.features,
                                     params=self.args.__dict__,
                                     embedding_size=self.args.embedding_size,
                                     embedding_l2_reg=self.args.embedding_l2_reg,
                                     embedding_dropout=self.args.embedding_dropout,
                                     dnn_hidden_units=self.args.dnn_hidden_units,
                                     dnn_activation=self.args.dnn_activation,
                                     dnn_l2_reg=self.args.dnn_l2_reg,
                                     dnn_use_bn=self.args.dnn_use_bn,
                                     dnn_dropout=self.args.dnn_dropout,
                                     init_std=self.args.init_std,
                                     seed=self.args.seed, )
        model = memonet_model.get_model()
        # optimizer & loss & metrics
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.args.learning_rate, beta_1=0.9, beta_2=0.999,
                                             epsilon=1e-8)
        loss = tf.keras.losses.BinaryCrossentropy()
        metrics = ["AUC", "binary_crossentropy"]
        # Model compile
        model.compile(optimizer, loss, metrics=metrics, )
        # model.run_eagerly = True
        # Print Info
        model.summary()
        keras.utils.plot_model(model, os.path.join(self.model_file_dir, "memonet.png"), show_shapes=True,
                               show_layer_names=True)
        return model


if __name__ == "__main__":
    runner = MemoNetRunner()
    runner.run()
