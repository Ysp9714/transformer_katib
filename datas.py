from pydantic import BaseModel
from itertools import cycle
import numpy as np
import tensorflow as tf
import math
from pathlib import Path
import pickle
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)  # suppress warnings


data_labels = ["long", "short"]
LONG_FEATURE = "long_features"
SHORT_FEATURE = "short_features"

class TrainingDataset(BaseModel):
    train_dataset: tf.data.Dataset
    val_dataset: tf.data.Dataset

    class Config:
        arbitrary_types_allowed = True


def fit_data_rate(datas):
    # datas = [long,short,label]
    # init
    add_list = [[] for i in range(len(datas))]
    result = []

    # Label Data
    label = datas[-1]
    label, count = np.unique(label, return_counts=True)
    label = list(map(int, label))
    max_num = max(count)
    
    non_max_list = list(count)
    while True:
        try:
            non_max_list.remove(max_num)
        except:
            break
    non_max_list = np.array(non_max_list)
    if all(non_max_list/max_num < 0.5):
        target_num = np.array([1,0.5,0.5]) * max_num
    else:
        max_num = max(non_max_list) * 2
        target_num = np.array([1,0.5,0.5]) * max_num
    # fit the data rate
    for data in cycle(zip(*datas)):
        if all(target_num == count):
            result = [
                tf.concat((d, tf.convert_to_tensor(r)), axis=0)
                for r, d in zip(add_list, datas)
            ]
            break
        if count[int(data[-1])] < target_num[int(data[-1])]:
            for ele, r in zip(data, add_list):
                r.append(ele)
            count[int(data[-1])] += 1
    return result, label


def seperate_data(datas, val_ratio) -> TrainingDataset:
    datas, labels = datas[:-1], datas[-1]
    train_size = math.ceil((1 - val_ratio) * len(labels))
    val_size = len(labels) - train_size

    # Preprocessing
    dataset = tf.data.Dataset.from_tensor_slices(
        ({key: val for key, val in zip(data_labels, datas)}, {"truth": labels})
    )
    dataset = dataset.shuffle(buffer_size=len(labels))

    # Train
    train_dataset = dataset.take(train_size)

    # Validation
    val_dataset = dataset.skip(train_size)
    return TrainingDataset(train_dataset=train_dataset, val_dataset=val_dataset)


def load_data(data_path, feature_num, label_num, feature_type="all") -> TrainingDataset:
    # read csv
    
    data_path = Path(data_path)
    long_feature = data_path / LONG_FEATURE
    short_feature = data_path / SHORT_FEATURE
    if not (
        long_feature.with_suffix('.pickle').exists()
        and short_feature.with_suffix('.pickle').exists()
    ):
        long_train_datas = np.loadtxt(
            fname=long_feature.with_suffix('.csv'), dtype=np.float32, delimiter=","
        )
        short_train_datas = np.loadtxt(
            fname=short_feature.with_suffix('.csv'), dtype=np.float32, delimiter=","
        )
        with open(long_feature.with_suffix('.pickle'), 'wb') as f:
            pickle.dump(long_train_datas,f)
        short_train_datas = np.loadtxt(
            fname=short_feature.with_suffix('.csv'), dtype=np.float32, delimiter=","
        )
        with open(short_feature.with_suffix('.pickle'), 'wb') as f:
            pickle.dump(short_train_datas,f)
    with open(long_feature.with_suffix('.pickle'), 'rb') as f:
        long_train_datas = pickle.load(f)
    with open(short_feature.with_suffix('.pickle'), 'rb') as f:
        short_train_datas = pickle.load(f)


    # separate into data and label
    long_datas, label_datas = long_train_datas[:, :-1], long_train_datas[:, -1]
    short_datas = short_train_datas[:, :-1]
    short_datas = short_datas[:, :feature_num]

    label_datas = [min(i, label_num) for i in label_datas]

    datas, label = fit_data_rate((long_datas, short_datas, label_datas))
    datasets = seperate_data(datas, 0.1)
    # return datas
    return TrainingDataset(
        train_dataset=datasets.train_dataset, val_dataset=datasets.val_dataset,
    )


if __name__ == "__main__":
    load_data("datas", 18, 3)
