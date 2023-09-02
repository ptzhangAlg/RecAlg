import json
import os
from collections import OrderedDict

from rec_alg.common.data_loader import DataLoader
from rec_alg.preprocessing.base_process import BaseProcess


class SparseProcess(BaseProcess):
    """
    Preprocessing for categorical fields in datasets
    """
    
    def __init__(self, config_path, **kwargs):
        super(SparseProcess, self).__init__(config_path=config_path, **kwargs)
        self.label_encoder = OrderedDict()
        self.label_counter = OrderedDict()
        self.sparse_path = self.config.get("base_info", {}).get("sparse_path", None)
        self.target_config_path = "{dir}/{name}".format(dir=os.path.dirname(self.config_path),
                                                        name="config_sparse.json")
        self._init()
        return
    
    def _init(self):
        DataLoader.validate_or_create_dir(self.sparse_path)
        return
    
    def fit(self, sep="\t", min_occurrences=10, index_base=0):
        """
        Encode categorical fields
        For feature appearing less or equal than min_occurrences, encode to 0. And feature appearing great than
        min_occurrences, encode to independent value.
        :param sep: separator of different fields
        :param min_occurrences: min occurrences of a feature
        :param index_base: zero-based or one-based
        :return:
        """
        cnt_train = 0
        fea_freqs = OrderedDict()
        for feature_index in self.sparse_feature_indexes:
            fea_freqs.setdefault(feature_index, OrderedDict())
        for file_path in self.train_files:
            with open(file_path, mode="r", encoding="utf-8") as fi:
                for line in fi:
                    items = line.strip('\n').split(sep=sep)
                    if len(items) == 0:
                        continue
            
                    cnt_train += 1
                    if cnt_train % 10000 == 0:
                        print('SparseProcess::fit: now train cnt : %d' % cnt_train)
            
                    for feature_index in self.sparse_feature_indexes:
                        cur_freqs = fea_freqs[feature_index]
                        item = items[feature_index]
                        if item not in cur_freqs:
                            cur_freqs[item] = 1
                        else:
                            cur_freqs[item] += 1

        print("get index in field")
        label_encoder = OrderedDict()
        label_counter = OrderedDict()
        for feature_index in self.sparse_feature_indexes:
            label_encoder.setdefault(feature_index, OrderedDict())
            label_counter.setdefault(feature_index, index_base)
        for index, cur_freqs in fea_freqs.items():
            current_encoder = label_encoder[index]
            min_occurrences_index = -1
            label_index = index_base
            for item, freq in cur_freqs.items():
                if freq <= min_occurrences:
                    if min_occurrences_index < 0:
                        min_occurrences_index = label_index
                        label_index += 1
                    current_encoder[item] = [min_occurrences_index, freq]
                else:
                    current_encoder[item] = [label_index, freq]
                    label_index += 1
            label_counter[index] = label_index

        print('SparseProcess::fit: number of categorical fields：', len(self.sparse_feature_indexes))
        print('SparseProcess::fit: number of features of every fields：{}'.format(label_counter))
        print('SparseProcess::fit: number of all the features: {}'.format(sum(label_counter.values())))
        print('SparseProcess::fit: total entries :%d' % cnt_train)
        for feature_index in self.sparse_feature_indexes:
            if feature_index != 1:
                continue
            print("Sparse feature {}: ".format(feature_index))
            for key, value in label_encoder[feature_index].items():
                print("key:{}\tindex:{}\tcount:{}".format(key, value[0], value[1]))

        self.label_encoder = label_encoder
        self.label_counter = label_counter

        self._update_config()
        return True
    
    def transform(self, sep="\t"):
        sparse_feature_indexes = set(self.sparse_feature_indexes)
        cnt_train = 0
        for file_path in self.train_files:
            fi = open(file_path, 'r', encoding="utf-8")
            fo = open(self.sparse_path + os.path.basename(file_path), 'w', encoding="utf-8")
            print('SparseProcess::transform: remake training data...')
            for line in fi:
                cnt_train += 1
                if cnt_train % 10000 == 0:
                    print('SparseProcess::transform: now train cnt : %d' % cnt_train)
                entry = []
                items = line.strip('\n').split(sep=sep)
                for index, item in enumerate(items):
                    if index in sparse_feature_indexes:
                        entry.append(str(self.label_encoder[index][item][0]))
                    else:
                        entry.append(item)
                fo.write(sep.join(entry) + '\n')
            fo.close()
            fi.close()
        return True
    
    def _update_config(self):
        for index, feature in enumerate(self.config["features"]):
            if feature["type"] == "sparse":
                feature["dimension"] = self.label_counter[index]
        
        self.config["base_info"]["train_path"] = self.config["base_info"]["sparse_path"]
        
        with open(self.target_config_path, 'w', encoding='utf-8') as json_file:
            json.dump(self.config, json_file, ensure_ascii=False, indent=4)
        return True
