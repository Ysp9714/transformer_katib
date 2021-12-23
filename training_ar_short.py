import logging
from operator import truth
import time
from itertools import chain

import tensorflow_datasets as tfds
import tensorflow_text as text
import tensorflow as tf

from model import Transformer
from utils import create_padding_mask, create_look_ahead_mask
from datas import load_data
from augmentation import change_data_shape, cut_serial_frame, label_to_onehot_v2
from parameter import get_arg


# suppress warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)


EPOCHS = 0

param = get_arg()
param.label = [0, 1, 2]
BUFFER_SIZE = 20000
BATCH_SIZE = 32

# Hyper Parameter
num_layers = 6
d_model = 128
dff = 512
num_heads = 8
dropout_rate = 0.1
target_size = len(param.label)


def make_batches(ds: tf.data.Dataset):
    return(
        ds
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
        .map(change_data_shape,)
        .cache()
        .map(cut_serial_frame, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
def make_val_batches(ds: tf.data.Dataset):
    return(
        ds
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
        .map(change_data_shape, num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        # .map(cut_serial_frame, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=2000) -> None:
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        # rsqrt == -0.5, sqrt == 2
        # rsqrt는 sqrt의 역수
        arg1 = tf.math.rsqrt(step)/2
        arg2 = step * (self.warmup_steps ** -1.5)
        rn = tf.math.rsqrt(self.d_model) * tf.math.minimum(tf.math.minimum(arg1, arg2),tf.constant(0.006,tf.float32))
        # print(step,rn)
        return rn / 2


learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(
    learning_rate=learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)


loss_object = tf.keras.losses.CategoricalCrossentropy(
    from_logits=False, reduction='none', label_smoothing=0.1)


def loss_function(real, pred):
    # 0인 값을 0, 그 외 값은 1로만들어 마스크를 생성
    # 예측되지 않은 부분이 loss에 포함되는 것을 막는다.
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    mask = tf.ones(tf.shape(real))

    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    mask = mask[:,:,0] * [0.8,0.2]
    loss_ *= mask
    num_total = 1
    for i in tf.shape(loss_):
        num_total *= i

    return tf.reduce_sum(loss_)/ tf.cast(num_total, tf.float32)


def accuracy_function(real, pred):
    # (batch_size, seq_len, d_model)
    accuracies = tf.equal(tf.argmax(real, axis=2), tf.argmax(pred, axis=2))
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    num_total = 1
    for i in tf.shape(accuracies):
        num_total *= i
    return tf.reduce_sum(accuracies) / tf.cast(num_total, tf.float32)


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')


# transformer = Transformer(
#     num_layers=num_layers,
#     d_model=d_model,
#     num_heads=num_heads,
#     dff=dff,
#     target_size=target_size,
#     pe_input=1000,
#     pe_target=1000,
#     rate=dropout_rate)


strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    transformer = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        target_size=target_size,
        pe_input=1000,
        pe_target=1000,
        rate=dropout_rate)

def create_masks(inp, tar):
    # Encoder padding mask
    inp_reduce = tf.reduce_sum(inp,-1)
    tar_reduce = tf.ones(tf.shape(tf.reduce_sum(tar,-1)))
    enc_padding_mask = create_padding_mask(inp_reduce)

    # Used in the 2nd attention block in the decoder
    # This pading mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp_reduce)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input recieved by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar_reduce)[1])
    dec_target_padding_mask = create_padding_mask(tar_reduce, padding=True)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


checkpoint_path = "./checkpoints/train"

ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# # if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print('Latest checkpoint restored!!',ckpt_manager.latest_checkpoint)


# The @tf.function trace-compiles train_step into a TF graph for faster
# execution. The function specializes to the precise shape of the argument
# tensors. To avoid re-tracing due to the variable sequence lengths or variable
# batch sizes (the last batch is smaller), use input_signature to specift
# more generic shapes.

# train_step_signature = [
#     tf.TensorSpec(shape=(None, None, param.feature_num), dtype=tf.float32),
#     tf.TensorSpec(shape=(None, None, 3), dtype=tf.int32),
# ]


# @tf.function(input_signature=train_step_signature)
@tf.function
def train_step(inp, tar):
    tar_inp = tar[:,:-1,:]
    tar_real = tar[:,1:,:]
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
        inp, tar_inp)
    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp,
                                     True,
                                     enc_padding_mask,
                                     combined_mask,
                                     dec_padding_mask)
        loss = loss_function(tar_real, predictions)
        # loss = loss_object(tar_real, predictions)
    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(accuracy_function(tar[:,1::,:], predictions))


def train(train_batches, val_batches):
    best_val_acc = 0.9
    for epoch in range(EPOCHS):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()

        for (batch, (inp, tar)) in enumerate(train_batches):
            train_step(inp['long'], tar['truth'])

            if batch % 10 == 0:
                print(
              f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

        # if (epoch + 1) % 5 == 0:
        #     ckpt_save_path = ckpt_manager.save()
        #     print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')

        val_result = [val_test(inp['long'], truth['truth']) for inp, truth in val_batches]
        val_acc = sum(val_result)/ len(val_result)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_save_path = ckpt_manager.save()
            print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')
        print(
          f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f} Val Accuracy {val_acc:.4f}')

        print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')


def val_test(inp, truth):
    output = tf.zeros([tf.shape(inp)[0],3])
    output = tf.expand_dims(output,1)
    for i in range(2):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
        inp, output)
        predictions, attention_weights = transformer(inp,
                                                     output,
                                                     False,
                                                     enc_padding_mask,
                                                     combined_mask,
                                                     dec_padding_mask)
        predictions = predictions[:,-1:,:]
        output = tf.concat([output,predictions], axis=1)

    output = tf.argmax(output[:,2,:],axis=-1)
    truth = tf.argmax(truth[:,2,:], axis=-1)
    acc = tf.cast(tf.equal(truth, output), tf.float32)
    return tf.reduce_sum(acc) / tf.cast(tf.shape(output)[0], tf.float32)

@tf.function
def evaluate(inp):
    output = tf.zeros([tf.shape(inp)[0],3])
    output = tf.expand_dims(output,1)
    # inp sentence is portuguese, hence adding the start and end token
    # as the target is english, the first word to the transformer should be the
    # english start token.
    # 빈 문장을 입력하면 0번째 인덱스는 start, -1번째 인덱스는 end값을 갖는다.
    # 배치 추가

    # predictions.shape == (batch_size, seq_len, vocab_size)
    for i in range(2):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
        inp, output)
        predictions, attention_weights = transformer(inp,
                                                 output,
                                                 False,
                                                 enc_padding_mask,
                                                 combined_mask,
                                                 dec_padding_mask)
        predictions = predictions[:,-1:,:]
        output = tf.concat([output,predictions], axis=1)

    return output, predictions


def print_translation(sentence, tokens, ground_truth):
    print(f'{"Input:":15s}: {sentence}')
    print(f'{"Prediction":15s}: {tokens.numpy().decode("utf-8")}')
    print(f'{"Ground truth":15s}: {ground_truth}')


if __name__ == "__main__":
    import numpy as np
    dataset = load_data('datas/0311', 18, 3)
    ############################
    train_batches = make_batches(dataset.train_dataset)
    BATCH_SIZE = 128
    val_batches = make_val_batches(dataset.val_dataset)
    train(train_batches, val_batches)
    train_batches = make_val_batches(dataset.train_dataset)
    test_result = {}
    cnt = 0
    for t in chain(val_batches, train_batches):
        output, weight = evaluate(t[0]['long'])
        gt = tf.argmax(t[1]['truth'], axis=-1)[:,-1]
        output_truth = tf.argmax(output[:,-1,:], axis=-1)
        comp = tf.equal(gt, output_truth)
        result = tf.reduce_sum(tf.cast(comp, tf.float32)) / tf.cast(tf.shape(comp)[0], tf.float32)

        # for o,t,c,w in zip(output, truth, comp, weight):
        #     cnt +=1
        #     if c[0] == False:
        #         test_result[str(w.numpy())[:10]]=[o.numpy(),t.numpy(),c.numpy(),w.numpy()]
        for o,t,c,w,out in zip(output_truth, gt, comp, weight, output):
            if (tf.minimum(tf.argmax(out[1]),1).numpy() != tf.argmax(out[2]).numpy()):
                print('not same: ',"truth: ",t.numpy(), tf.minimum(tf.argmax(out[1]),1).numpy(), "prediction: ",tf.argmax(out[2]).numpy())
                print(out)
            test_result[str(w.numpy())]=[o.numpy(),t.numpy(),c.numpy(),w.numpy()]
    test_result