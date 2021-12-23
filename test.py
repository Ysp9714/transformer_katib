import logging
from pathlib import Path
import tensorflow as tf
import numpy as np

from model import Transformer
from utils import create_padding_mask, create_look_ahead_mask


# suppress warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)


def make_model(ckpt_path='checkpoints/train/saved_model.pb'):
    ckpt_path = str(ckpt_path)
    transformer = Transformer(
        num_layers=6,
        d_model=128,
        num_heads=8,
        dff=512,
        target_size=3,
        pe_input=1000,
        pe_target=1000,
        rate=0.1)
    ckpt = tf.train.Checkpoint(transformer=transformer)
    # # if a checkpoint exists, restore the latest checkpoint.
    ckpt.restore(ckpt_path)
    return transformer

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


@tf.function
def evaluate(model,inp):
    output = tf.zeros([tf.shape(inp)[0],3])
    output = tf.expand_dims(output,1)

    for _ in range(2):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
        inp, output)
        predictions, _ = model(inp,
                                output,
                                False,
                                enc_padding_mask,
                                combined_mask,
                                dec_padding_mask)
        predictions = predictions[:,-1:,:]
        output = tf.concat([output,predictions], axis=1)

    return tf.squeeze(predictions)


if __name__ == "__main__":
    model_path = Path(__file__).parent/'checkpoints/train/saved_model.pb'
    transformer = make_model(model_path)
    a = np.load('test_data.npy')
    prediction = evaluate(transformer,a)
    print(prediction)
    print()