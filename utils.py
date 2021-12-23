import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 제발 numpy, tensor를 다룰때는 값이 하나가 아닌 여러개의 값(Matrix)이라는 것을 인지하고 코딩하자
# 물론 아래 함수들은 하나의 값이여도 동작을 하지만, matrix값이 입력으로 들어와도 동작한다.
# 하나의 값이 들어갔을때 동작하게 함수를 작성하되 작성된 함수가 매트릭스연산에 사용되어도 이상함이 없어야한다.
# 오히러 매트릭스 연산에 동작하지 않음은 잘못된 함수이다.
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    # d_model은 모델의 depth
    angle_rads = get_angles(
        np.arange(position)[:,np.newaxis],
        np.arange(d_model)[np.newaxis,:],
        d_model)
    
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos in odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    # 차원을 추가하는 이유가 뭘까.. 1개의 batch사이즈를 가진 하나의 포지셔널인코딩 값이라는 것인가..
    return tf.cast(pos_encoding, dtype=tf.float32)

def pos_encoding_graph():
    n,d = 2048, 512
    pos_encoding = positional_encoding(n,d)
    print(pos_encoding.shape)
    pos_encoding = pos_encoding[0]

    # Juggle the dimensions for the plot
    pos_encoding = tf.reshape(pos_encoding, (n,d//2,2))
    pos_encoding = tf.transpose(pos_encoding, (2,1,0))
    pos_encoding = tf.reshape(pos_encoding, (d,n))

    plt.pcolormesh(pos_encoding, cmap='RdBu')
    plt.ylabel('Depth')
    plt.xlabel('Position')
    plt.colorbar()
    plt.show()

def create_padding_mask(seq, padding=False):
    seq = tf.cast(seq,tf.float32)
    if padding is False:
        seq += tf.cast(tf.constant([1e-10]), tf.float32)
    mask = tf.cast(tf.math.equal(seq,0), tf.float32)
    # add extra dimensions to add the padding
    # to the attention logits.
    mask = tf.expand_dims(mask,1)
    mask = tf.expand_dims(mask,1)
    return mask
    # return mask[:, tf.newaxis, tf.newaxis, :] # (batch,1,1, seq_len)

def create_look_ahead_mask(size):
    """
    1인 값은 추후 *-1e9(음의 무한대)값을 곱한 후 더하게 된다.
    즉 음의 무한대의 값을 갖게 되어 그 값은 소프트 맥스를 통과하게 되면 확률값이 0으로 수렴한다.
    Returns:
        e.g 
        size: 3
        <tf.Tensor: shape=(3, 3), dtype=float32, numpy=
        array([[0., 1., 1.],
               [0., 0., 1.],
               [0., 0., 0.]], dtype=float32)>
    """
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1,0)
    return mask # (seq_len, seq_len)