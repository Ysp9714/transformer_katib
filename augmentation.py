import sys
from pathlib import Path
# sys.path.append((Path(__file__).parent).parent)
# print((Path(__file__).parent).parent)
import random
import tensorflow as tf
from models import HyperParameter


def zero_to_small_num(short):
    return tf.maximum(1e-8,short)

def cut_frames(dataset, num):
    results = []
    for frames in dataset[0]['long']:
        cut_frame = frames[num:,:]
        result = tf.concat([cut_frame,tf.zeros((num,18,1))], axis=0)
        results.append(result)
    tf_results = tf.stack(results)
    return {'long':tf_results, 'short':dataset[0]['short']}, {'truth':dataset[1]['truth']}

def label_to_onehot(y):
    param = HyperParameter()
    y = tf.convert_to_tensor(y)
    y = y[...,tf.newaxis]
    onehot = tf.cast(param.label == y, tf.uint8)
    return onehot


def label_to_onehot_v2(y):
    y = tf.cast(tf.convert_to_tensor(y), tf.int32)
    y = y[...,tf.newaxis]
    label = tf.cast(tf.convert_to_tensor([0,1,2]), tf.int32)
    first_onehot = tf.cast(label == y, tf.int32)
    second_onehot = tf.cast(label == tf.minimum(y,1), tf.int32)
    init_onehot = first_onehot * 0
    return tf.stack([init_onehot ,first_onehot,second_onehot],axis=1)


def random_add_value(train_data,y):
    param = HyperParameter()
    # random_val = tf.random.uniform([frame_num,feature_num,1], minval=0.0, maxval=0.5, dtype=tf.float32)
    random_val = tf.random.normal([param.frame_num,param.feature_num,1], mean=0, stddev=0.1, dtype=tf.float32)
    if tf.random.uniform([1], minval=0, maxval=1) > 0.8:
        train_data['long'] = train_data['long'] + random_val
    return train_data, y


def change_data_shape(train_data, y):
    param = HyperParameter()
    long_data = train_data['long'] + 1e-5
    batch_size = tf.shape(train_data['long'])[0]
    long_data = tf.reshape(long_data,(batch_size, -1, param.feature_num))
    short_data = train_data['short'][...,tf.newaxis]
    truth = label_to_onehot_v2(y['truth'])
    return {'long':long_data, 'short':short_data}, {'truth':truth}


@tf.function
def cut_random_frame(train_data,y):
    param = HyperParameter()
    batch = train_data['long']
    def select_frame(frames):
        result = []
        random_frame_nums = random.choices(list(range(param.total_frame_num)),k=param.frame_num)
        random_frame_nums.sort()
        for frame_num in random_frame_nums:
            result.append(frames[frame_num])
        tf_result = tf.stack(result)
        return tf_result
    result = tf.map_fn(select_frame, batch)
    return {'long':result, 'short':train_data['short']},y


@tf.function
def cut_serial_frame(train_data,y):
    num_rand = tf.random.uniform([],0,1)
    param = HyperParameter()
    cut_size = param.total_frame_num - param.frame_num
    result = None
    long_data = train_data["long"]
    def cut_frame(long_data):
        start_num = tf.random.uniform([],0, param.total_frame_num-cut_size,dtype=tf.int64)
        front = long_data[:start_num,:]            
        back = long_data[start_num+cut_size:,:]
        data = tf.concat([front,back], axis=0)
        return data
    if num_rand > 0.5:
        result = tf.map_fn(cut_frame,long_data)
    else:
        result = long_data
    return {'long':result, 'short':train_data['short']},y