from rec_alg.preprocessing.dense_process import DenseProcess
from rec_alg.preprocessing.kdd12.raw_data_process import RawDataProcess
from rec_alg.preprocessing.kfold_process import KFoldProcess
from rec_alg.preprocessing.sparse_process import SparseProcess


def main():
    chunksize = 10000000
    # Process raw data to one file
    raw_data = RawDataProcess(config_path="./config/kdd12/config_template.json")
    raw_data.fit()
    raw_data.transform(sep="\t", chunksize=chunksize)
    
    # Preprocessing categorical fields：Fill null and LowFreq value with index-base, and encode label
    sparse = SparseProcess(config_path=raw_data.target_config_path)
    sparse.fit(min_occurrences=10, index_base=0)
    sparse.transform()

    # Preprocessing dense fields：Fill null with 0, and do scale-transform
    dense = DenseProcess(config_path=sparse.target_config_path)
    dense.fit(dense.scale_multi_min_max)
    dense.transform()

    # K-Fold
    kfold = KFoldProcess(config_path=dense.target_config_path)
    kfold.fit()
    kfold.transform(chunksize=chunksize, mode="fast")
    return


if __name__ == "__main__":
    main()
