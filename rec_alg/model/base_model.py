import os
import re

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from rec_alg.common.batch_generator import BatchGenerator
from rec_alg.common.constants import Constants
from rec_alg.common.data_loader import DataLoader
from rec_alg.components.inputs import SparseFeat, DenseFeat, VarLenSparseFeat


class BaseModel(object):
    """
    BaseModel
    """
    CHECKPOINT_TEMPLATE = "cp-{epoch:04d}.ckpt"
    CHECKPOINT_RE_TEMPLATE = "^cp-(.*).ckpt"
    
    def __init__(self, config_path, **kwargs):
        # Load config for config_path
        self.config = DataLoader.load_config_dict(config_path=config_path)
        self.model_suffix = kwargs.get("model_path", "")
        self.model_config = self.config.get("model", {})
        self.sequence_max_len = kwargs.get("sequence_max_len", 0)
        # Load dataset feature info
        self._load_feature_info()
        # Create dirs
        self._create_dirs()
        # Get input/output files
        self._get_input_output_files()
        
        self.args = getattr(self, "args", None)
        return
    
    def _load_feature_info(self):
        """
        Load feature info of a dataset from config
        :return:
        """
        self.features = []
        self.sparse_features = []
        self.dense_features = []
        self.varlen_features = []
        for feature in self.config["features"]:
            if feature["type"] == Constants.FEATURE_TYPE_SPARSE:
                sparse_feature = SparseFeat(name=feature["name"], dimension=feature["dimension"],
                                            use_hash=feature["use_hash"],
                                            dtype=feature["dtype"], embedding=feature["embedding"],
                                            embedding_name=feature.get("embedding_name", None),
                                            feature_num=feature.get("feature_num", None),
                                            feature_origin_num=feature.get("feature_origin_num", None),
                                            feature_info_gain=feature.get("feature_info_gain", None),
                                            feature_ig=feature.get("feature_ig", None),
                                            feature_attention=feature.get("feature_attention", None))
                self.sparse_features.append(sparse_feature)
                self.features.append(sparse_feature)
            elif feature["type"] == Constants.FEATURE_TYPE_DENSE:
                dense_feature = DenseFeat(name=feature["name"], dimension=feature.get("dimension", 1),
                                          dtype=feature["dtype"],
                                          feature_num=feature.get("feature_num", None),
                                          feature_origin_num=feature.get("feature_origin_num", None),
                                          feature_info_gain=feature.get("feature_info_gain", None),
                                          feature_ig=feature.get("feature_ig", None),
                                          feature_attention=feature.get("feature_attention", None))
                self.dense_features.append(dense_feature)
                self.features.append(dense_feature)
            elif feature["type"] == Constants.FEATURE_TYPE_VARLENSPARSE:
                varlen_feature = VarLenSparseFeat(
                    name=feature["name"], dimension=feature["dimension"],
                    maxlen=feature["maxlen"] if self.sequence_max_len <= 0 else self.sequence_max_len,
                    combiner=feature["combiner"],
                    use_hash=feature["use_hash"],
                    dtype=feature["dtype"], embedding=feature["embedding"],
                    embedding_name=feature.get("embedding_name", None),
                    feature_num=feature.get("feature_num", None),
                    feature_origin_num=feature.get("feature_origin_num", None),
                    feature_info_gain=feature.get("feature_info_gain", None),
                    feature_ig=feature.get("feature_ig", None),
                    feature_attention=feature.get("feature_attention", None))
                self.varlen_features.append(varlen_feature)
                self.features.append(varlen_feature)
        return True
    
    def _create_dirs(self):
        self.checkpoint_dir = os.path.join(self.model_config.get("results_prefix", "./data/model/"),
                                           self.args.model_path, "checkpoint")
        self.model_file_dir = os.path.join(self.model_config.get("model_file_path", "./data/model/"),
                                           self.args.model_path, "model")
        DataLoader.validate_or_create_dir(self.checkpoint_dir)
        DataLoader.validate_or_create_dir(self.model_file_dir)
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
    
    def create_model(self):
        """
        由子类创建具体的keras.model
        :return:
        """
        raise NotImplementedError()
    
    def create_checkpoint_callback(self, save_weights_only=True, period=1):
        """
        Create callback function of checkpoint
        :return: callback
        """
        checkpoint_path = "{checkpoint_dir}/{name}".format(checkpoint_dir=self.checkpoint_dir,
                                                           name=self.CHECKPOINT_TEMPLATE)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            verbose=1,
            save_weights_only=save_weights_only,
            period=period, )
        return cp_callback
    
    def create_earlystopping_callback(self, monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto',
                                      baseline=None, restore_best_weights=True):
        """
        Create early stopping callback
        :return: callback
        """
        es_callback = tf.keras.callbacks.EarlyStopping(monitor=monitor, min_delta=min_delta, patience=patience,
                                                       verbose=verbose, mode=mode, baseline=baseline,
                                                       restore_best_weights=restore_best_weights)
        return es_callback
    
    def restore_model_from_checkpoint(self, restore_epoch=-1):
        """
        Restore model weights from checkpoint created by restore_epoch
        Notice: these are only weights in checkpoint file, model structure should be created by create_model
        :param restore_epoch: from which checkpoint to load weights
        :return: model, latest_epoch
        """
        # Get checkpoint path
        latest_checkpoint_path = tf.train.latest_checkpoint(self.checkpoint_dir)
        latest_epoch = self._get_latest_epoch_from_checkpoint(latest_checkpoint_path)
        checkpoint_path = os.path.join(self.checkpoint_dir, self.CHECKPOINT_TEMPLATE.format(
            epoch=restore_epoch)) if 0 < restore_epoch <= latest_epoch else latest_checkpoint_path
        
        # create model and load weights from checkpoint
        model = self.create_model()
        model.load_weights(checkpoint_path).expect_partial()
        print("BaseModel::restore_model_from_checkpoint: restore model from checkpoint: ", checkpoint_path)
        return model, latest_epoch
    
    def _get_latest_epoch_from_checkpoint(self, latest_checkpoint):
        """
        Get latest epoch from checkpoint path
        :param latest_checkpoint:
        :return:
        """
        latest_epoch = 0
        regular = re.compile(self.CHECKPOINT_RE_TEMPLATE)
        try:
            checkpoint = os.path.basename(latest_checkpoint)
            match_result = regular.match(checkpoint)
            latest_epoch = int(match_result.group(1))
        except Exception as e:
            print(e)
        return latest_epoch
    
    def run(self):
        if self.args.mode in (Constants.MODE_TRAIN, Constants.MODE_RETRAIN):
            self.train_model()
            self.test_model()
        elif self.args.mode == Constants.MODE_TEST:
            self.test_model()
        return True
    
    def train_model(self):
        """
        Train model
        :return: history
        """
        if self.args.mode == Constants.MODE_RETRAIN:
            model, latest_epoch = self.restore_model_from_checkpoint()
        else:
            model = self.create_model()
            latest_epoch = 0
        callbacks = [self.create_checkpoint_callback(), self.create_earlystopping_callback(), ]
        
        # 1. Get data from generator (single process & thread)
        train_steps = BatchGenerator.get_txt_dataset_length(self.train_files, batch_size=self.args.batch_size,
                                                            drop_remainder=True)
        val_steps = BatchGenerator.get_txt_dataset_length(self.valid_files, batch_size=self.args.batch_size,
                                                          drop_remainder=False)
        train_generator = BatchGenerator.generate_arrays_from_file(self.train_files, batch_size=self.args.batch_size,
                                                                   drop_remainder=True, features=self.features,
                                                                   shuffle=True)
        val_generator = BatchGenerator.generate_arrays_from_file(self.valid_files, batch_size=self.args.batch_size,
                                                                 drop_remainder=False, features=self.features,
                                                                 shuffle=False)
        print("Train files: ", self.train_files)
        print("Train steps: ", train_steps)
        print("Valid files: ", self.valid_files)
        print("Valid steps: ", val_steps)
        
        history = model.fit_generator(train_generator, steps_per_epoch=train_steps, epochs=self.args.epochs,
                                      verbose=self.args.verbose, validation_data=val_generator,
                                      validation_steps=val_steps,
                                      callbacks=callbacks, max_queue_size=10, workers=1, use_multiprocessing=False,
                                      shuffle=False, initial_epoch=latest_epoch)
        self._save_train_results(latest_epoch, history)
        return history
    
    def _save_train_results(self, latest_epoch, history):
        df = pd.DataFrame(history.history)
        df.insert(0, "epoch", range(latest_epoch + 1, latest_epoch + len(df) + 1))
        if len(df) > 0:
            df.to_csv(self.train_results_file, sep="\t", float_format="%.5f", index=False, encoding="utf-8", mode="a")
        return
    
    def test_model(self):
        """
        Test FiBiNET model, support to test model from specific checkpoint
        :return:
        """
        restore_epochs = []
        if not isinstance(self.args.restore_epochs, list) or len(self.args.restore_epochs) == 0:
            restore_epochs = np.arange(1, self.args.epochs + 1)
        elif len(self.args.restore_epochs) == 1:
            restore_epochs = np.arange(1, self.args.restore_epochs[0])
        elif len(self.args.restore_epochs) == 2:
            restore_epochs = np.arange(self.args.restore_epochs[0], self.args.restore_epochs[1])
        elif len(self.args.restore_epochs) >= 3:
            restore_epochs = np.arange(self.args.restore_epochs[0], self.args.restore_epochs[1],
                                       self.args.restore_epochs[2])
        print("BaseModel::test_model: restore_epochs: {}".format(restore_epochs))
        for restore_epoch in restore_epochs:
            self.test_model_from_checkpoint(restore_epoch)
        return True
    
    def test_model_from_checkpoint(self, restore_epoch=-1):
        model, latest_epoch = self.restore_model_from_checkpoint(restore_epoch=restore_epoch)
        test_steps = BatchGenerator.get_dataset_length(self.test_files, batch_size=self.args.batch_size,
                                                       drop_remainder=False)
        test_generator = BatchGenerator.generate_arrays_from_file(self.test_files, batch_size=self.args.batch_size,
                                                                  features=self.features, drop_remainder=False,
                                                                  shuffle=False)
        print("Test files: ", self.test_files)
        print("Test steps: ", test_steps)
        predict_ans = model.evaluate_generator(test_generator, steps=test_steps, verbose=self.args.verbose)
        results_dict = dict(zip(model.metrics_names, predict_ans))
        print("BaseModel::test_model_from_checkpoint: Epoch {} Evaluation results: {}".format(restore_epoch,
                                                                                              results_dict))
        self._save_test_results(restore_epoch, results_dict)
        return
    
    def _save_test_results(self, restore_epoch, results_dict):
        df = pd.DataFrame(columns=results_dict.keys())
        df.loc[0] = list(results_dict.values())
        df.insert(0, "epoch", "{}".format(restore_epoch))
        if len(df) > 0:
            df.to_csv(self.test_results_file, sep="\t", float_format="%.5f", index=False, encoding="utf-8", mode="a")
        return
