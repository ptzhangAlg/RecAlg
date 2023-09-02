import json
import os

import numpy as np
import pandas as pd

from rec_alg.common.data_loader import DataLoader
from rec_alg.preprocessing.base_process import BaseProcess


class RawDataProcess(BaseProcess):
    """
    1. concat train.txt and userid_profile.txt to make a full train.txt file
    2. process label to 0 and 1
    3. Need huge memory
    """
    
    def __init__(self, config_path, ):
        super(RawDataProcess, self).__init__(config_path)
        self.concat_path = self.config.get("base_info", {}).get("concat_path", None)
        self.target_config_path = "{dir}/{name}".format(dir=os.path.dirname(self.config_path),
                                                        name="config_concat.json")
        
        self._init()
        return
    
    def _init(self):
        DataLoader.validate_or_create_dir(self.concat_path)
        return
    
    def fit(self):
        return
    
    def transform(self, sep="\t", chunksize=10000):
        # Load data
        df_user = DataLoader.load_data_txt_as_df(path=os.path.join(self.train_path, "userid_profile.txt"), sep="\t", )
        iterator = pd.read_csv(os.path.join(self.train_path, "training.txt"), sep=sep, header=None, index_col=None,
                               chunksize=chunksize, encoding="utf-8")
        target_file = os.path.join(self.concat_path, "train.txt")
        if os.path.exists(target_file):
            os.remove(target_file)

        for n, data_chunk in enumerate(iterator):
            print('RawDataProcess::transform: Size of uploaded chunk: %i instances, %i features' % data_chunk.shape)
            print("RawDataProcess::transform: chunk counter: {}".format(n))

            # concat data
            df_target = pd.merge(data_chunk, df_user, left_on=data_chunk.columns[-1], right_on=df_user.columns[0],
                                 how='left')
            df_target.drop(columns=df_target.columns[-len(df_user.columns)], inplace=True)

            # Missing value handling
            df_target.iloc[:, -2] = df_target.iloc[:, -2].apply(
                lambda x: 0 if x is None or x == "" or np.isnan(x) else x)
            df_target.iloc[:, -1] = df_target.iloc[:, -1].apply(
                lambda x: 0 if x is None or x == "" or np.isnan(x) else x)

            # Label
            df_target.iloc[:, 0] = df_target.iloc[:, 0].apply(lambda x: x if int(x) == 0 else 1)

            # write out
            df_target.to_csv(os.path.join(self.concat_path, "train.txt"), sep='\t', header=None, index=False, mode="a")

        self._update_config()
        pass
    
    def _update_config(self):
        self.config["base_info"]["train_path"] = self.concat_path
        with open(self.target_config_path, 'w', encoding='utf-8') as json_file:
            json.dump(self.config, json_file, ensure_ascii=False, indent=4)
        return True
