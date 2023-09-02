import argparse
import os

import tensorflow as tf
from tensorflow import keras

from rec_alg.common.data_loader import DataLoader
from rec_alg.common.utils import Utils
from rec_alg.model.base_model import BaseModel
from rec_alg.model.fibinet.fibinet_model import FiBiNetModel


# tf.enable_eager_execution()


class FiBiNetRunner(BaseModel):
    """
    train & test FiBiNetModel with supported args
    """
    CHECKPOINT_TEMPLATE = "cp-{epoch:04d}.ckpt"
    CHECKPOINT_RE_TEMPLATE = "^cp-(.*).ckpt"
    
    def __init__(self):
        # Get args
        self.args = self.parse_args()
        self._update_parameters()
        print("Args: ", self.args.__dict__)
        args_dict = self.args.__dict__.copy()
        config_path = args_dict.pop("config_path")
        super(FiBiNetRunner, self).__init__(config_path=config_path, **args_dict)
        # Get input/output files
        self._get_input_output_files()
        return
    
    def _update_parameters(self):
        if self.args.version == "v1":
            parameters = {
                "sparse_embedding_norm_type": "none",
                "dense_embedding_norm_type": "none",
                "senet_squeeze_mode": "mean",
                "senet_squeeze_group_num": 1,
                "senet_excitation_mode": "vector",
                "senet_activation": "relu",
                "senet_use_skip_connection": False,
                "senet_reweight_norm_type": "none",
                "origin_bilinear_type": "all",
                "origin_bilinear_dnn_units": [],
                "origin_bilinear_dnn_activation": "linear",
                "senet_bilinear_type": "all",
                "enable_linear": True
            }
            self.args.__dict__.update(parameters)
        elif self.args.version == "++":
            parameters = {
                "sparse_embedding_norm_type": "bn",
                "dense_embedding_norm_type": "layer_norm",
                "senet_squeeze_mode": "group_mean_max",
                "senet_squeeze_group_num": 2,
                "senet_excitation_mode": "bit",
                "senet_activation": "none",
                "senet_use_skip_connection": True,
                "senet_reweight_norm_type": "ln",
                "origin_bilinear_type": "all_ip",
                "origin_bilinear_dnn_units": [50],
                "origin_bilinear_dnn_activation": "linear",
                "senet_bilinear_type": "none",
                "enable_linear": False,
            }
            self.args.__dict__.update(parameters)
        return
    
    def _get_input_output_files(self):
        train_paths = self.args.train_paths if self.args.train_paths else self.model_config["train_paths"]
        valid_paths = self.args.valid_paths if self.args.valid_paths else self.model_config["valid_paths"]
        test_paths = self.args.test_paths if self.args.test_paths else self.model_config["test_paths"]
        print("Train paths: ", self.model_config["data_prefix"], train_paths)
        print("Valid paths: ", self.model_config["data_prefix"], valid_paths)
        print("Test paths: ", self.model_config["data_prefix"], test_paths)
        self.train_files = DataLoader.get_files(self.model_config["data_prefix"], train_paths)
        self.valid_files = DataLoader.get_files(self.model_config["data_prefix"], valid_paths)
        self.test_files = DataLoader.get_files(self.model_config["data_prefix"], test_paths)
        self.train_results_file = os.path.join(self.model_config["results_prefix"],
                                               self.args.model_path,
                                               self.model_config["train_results_file"])
        self.test_results_file = os.path.join(self.model_config["results_prefix"],
                                              self.args.model_path,
                                              self.model_config["test_results_file"])
        return
    
    def parse_args(self):
        parser = argparse.ArgumentParser()
        # 1. Run setup
        parser.add_argument('--version', type=str, default="++",
                            help="#version: version of fibinet model, support v1, ++ and custom")
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
        # 2.1. Embeddings
        parser.add_argument('--embedding_size', type=int, default=10,
                            help='#embedding_size: feature embedding size')
        parser.add_argument('--embedding_l2_reg', type=float, default=0.0,
                            help='#embedding_l2_reg: L2 regularizer strength applied to embedding')
        parser.add_argument('--embedding_dropout', type=float, default=0.0,
                            help='#embedding_dropout: the probability of dropping out on embedding')
        parser.add_argument('--sparse_embedding_norm_type', type=str, default='bn',
                            help='#sparse_embedding_norm_type: str, support `none, bn')
        parser.add_argument('--dense_embedding_norm_type', type=str, default='layer_norm',
                            help='#dense_embedding_norm_type: str, support `none, layer_norm')
        parser.add_argument('--dense_embedding_share_params', type=Utils.str2bool, default=False,
                            help='#dense_embedding_share_params: whether sharing params among different fields')
        
        # 2.2. SENet
        parser.add_argument('--senet_squeeze_mode', type=str, default='group_mean_max',
                            help='#senet_squeeze_mode: mean, max, topk, and group')
        parser.add_argument('--senet_squeeze_group_num', type=int, default=2,
                            help='#senet_squeeze_group_num: worked only in group mode')
        parser.add_argument('--senet_squeeze_topk', type=int, default=1,
                            help='#senet_squeeze_topk: positive integer, topk value')
        parser.add_argument('--senet_reduction_ratio', type=float, default=3.0,
                            help='#senet_reduction_ratio: senet reduction ratio')
        parser.add_argument('--senet_excitation_mode', type=str, default="bit",
                            help='#senet_excitation_mode: str, support: none(=squeeze_mode), vector|group|bit')
        parser.add_argument('--senet_activation', type=str, default='none',
                            help='#senet_activation: activation function used in SENet Layer 2')
        parser.add_argument('--senet_use_skip_connection', type=Utils.str2bool, default=True,
                            help='#senet_use_skip_connection:  bool.')
        parser.add_argument('--senet_reweight_norm_type', type=str, default='ln',
                            help='#senet_reweight_norm_type: none, ln')
        
        # 2.3. Bilinear type
        parser.add_argument('--origin_bilinear_type', type=str, default='all_ip',
                            help='#origin_bilinear_type: bilinear type applied to original embeddings')
        parser.add_argument('--origin_bilinear_dnn_units', type=Utils.str2list, default=[50],
                            help='#origin_bilinear_dnn_units: list')
        parser.add_argument('--origin_bilinear_dnn_activation', type=str, default='linear',
                            help='#origin_bilinear_dnn_activation: Activation function to use in DNN')
        parser.add_argument('--senet_bilinear_type', type=str, default='none',
                            help='#senet_bilinear_type: bilinear type applied to senet embeddings')
        
        # 2.4. DNN part
        parser.add_argument('--dnn_hidden_units', type=Utils.str2list, default=[400, 400, 400],
                            help='#dnn_hidden_units: layer number and units in each layer of DNN')
        parser.add_argument('--dnn_activation', type=str, default='relu',
                            help='#dnn_activation: activation function used in DNN')
        parser.add_argument('--dnn_l2_reg', type=float, default=0.0,
                            help='#dnn_l2_reg: L2 regularizer strength applied to DNN')
        parser.add_argument('--dnn_use_bn', type=Utils.str2bool, default=False,
                            help='#dnn_use_bn: whether to use BatchNormalization before activation or not in DNN')
        parser.add_argument('--dnn_dropout', type=float, default=0.0,
                            help='#dnn_dropout: the probability of dropping out on each layer of DNN')
        
        # 2.5. Linear part
        parser.add_argument('--enable_linear', type=Utils.str2bool, default=False,
                            help='#enable_linear:  bool. Whether use linear part in the model')
        parser.add_argument('--linear_l2_reg', type=float, default=0.0,
                            help='#linear_l2_reg: L2 regularizer strength applied to linear')
        
        # 3. Train/Valid/Test setup
        parser.add_argument('--seed', type=int, default=1024, help='#seed: integer ,to use as random seed.')
        parser.add_argument('--epochs', type=int, default=5)
        parser.add_argument('--batch_size', type=int, default=1024)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        parser.add_argument('--init_std', type=float, default=0.01)
        parser.add_argument('--verbose', type=int, default=1)
        parser.add_argument("--mode", type=str, default="train", help="support: train, retrain, test")
        parser.add_argument('--restore_epochs', type=Utils.str2list, default=[],
                            help="restore weights from checkpoint, format like np.arange(), eg. [1, 5, 1]")
        parser.add_argument("--early_stopping", type=Utils.str2bool, default=True, help="enable early stopping")
        parser.add_argument("--model_path", type=str, default="fibinet", help="model_path, to avoid being covered")
        return parser.parse_args()
    
    def create_model(self):
        """
        Create FiBiNet model
        :return: instance of FiBiNet model: tf.keras.Model
        """
        fibinet = FiBiNetModel(params=self.args.__dict__,
                               feature_columns=self.features,
                               embedding_size=self.args.embedding_size,
                               embedding_l2_reg=self.args.embedding_l2_reg,
                               embedding_dropout=self.args.embedding_dropout,
                               sparse_embedding_norm_type=self.args.sparse_embedding_norm_type,
                               dense_embedding_norm_type=self.args.dense_embedding_norm_type,
                               dense_embedding_share_params=self.args.dense_embedding_share_params,
                               senet_squeeze_mode=self.args.senet_squeeze_mode,
                               senet_squeeze_group_num=self.args.senet_squeeze_group_num,
                               senet_squeeze_topk=self.args.senet_squeeze_topk,
                               senet_reduction_ratio=self.args.senet_reduction_ratio,
                               senet_excitation_mode=self.args.senet_excitation_mode,
                               senet_activation=self.args.senet_activation,
                               senet_use_skip_connection=self.args.senet_use_skip_connection,
                               senet_reweight_norm_type=self.args.senet_reweight_norm_type,
                               origin_bilinear_type=self.args.origin_bilinear_type,
                               origin_bilinear_dnn_units=self.args.origin_bilinear_dnn_units,
                               origin_bilinear_dnn_activation=self.args.origin_bilinear_dnn_activation,
                               senet_bilinear_type=self.args.senet_bilinear_type,
                               dnn_hidden_units=self.args.dnn_hidden_units,
                               dnn_activation=self.args.dnn_activation,
                               dnn_l2_reg=self.args.dnn_l2_reg,
                               dnn_use_bn=self.args.dnn_use_bn,
                               dnn_dropout=self.args.dnn_dropout,
                               enable_linear=self.args.enable_linear,
                               linear_l2_reg=self.args.linear_l2_reg,
                               init_std=self.args.init_std,
                               seed=self.args.seed, )
        model = fibinet.get_model()
        # optimizer & loss & metrics
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.args.learning_rate, beta_1=0.9, beta_2=0.999,
                                             epsilon=1e-8)
        loss = tf.keras.losses.BinaryCrossentropy()
        metrics = ["AUC", "binary_crossentropy"]
        model.compile(optimizer, loss, metrics=metrics)
        # model.run_eagerly = True
        # Print Info
        model.summary()
        tf.keras.utils.plot_model(model, os.path.join(self.model_file_dir, "fibinet.png"), show_shapes=True,
                                  show_layer_names=True)
        return model


if __name__ == "__main__":
    runner = FiBiNetRunner()
    runner.run()
    pass
