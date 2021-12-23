import tensorflow as tf
import numpy as np

# Scaled Dot-Product Attention
# 내적 관심도는 깊이의 제곱근 계수로 조정된다. 
# 분산값이 너무 낮으면 출력이 평평하여 일관된 분산을 얻을 수 없다.
# 분산값이 너무 크면 소프트맥스가 초기화시 포화되어 학습하기 어렵다.
# mask는 미래의 값에 -1e9가 곱해져 그 값을 거의 0에 수렴하도록 한다.
def scaled_dot_product_attention(q,k,v, mask=None):
    """
    Calculate the attention weights.
    q,k,v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for adition.
    Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth)
        mask: Float tensor with shape broadcastable
              to (..., seq_len_q, seq_len_k). Defaults to None.
    Returns:
        output, attention_weights
    """

    # Q * K^T (Dot Product)
    # multihead -> (batch_size, num_heads, seq_len, d_model//num_heads)
    matmul_qk = tf.matmul(q,k, transpose_b=True) # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    # dimension of key, not length. depth
    # \sqrt{d_K}
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # masked(opt)
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    
    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1) # (..., seq_len_q, seq_len_k)
    output = tf.matmul(attention_weights, v) # (..., seq_len_q, depth_v)
    return output, attention_weights
# 소프트 맥스 정규화가 K에서 수행되므로 해당 값은 Q에 부여되는 중요도를 결정합니다.
# 출력은 주의 가중치(attention_weights)와 V(값)벡터의 곱셈을 나타냅니다. 이렇게 하면 집중하려는 단어는 유지되고 관련 없는 단어는 제거됩니다.


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads
        
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """ Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        e.g
        x = <tf.Tensor: shape=(1, 4, 4), dtype=int32, numpy=
        array([[[1, 2, 3, 4],
                [5, 6, 7, 8],
                [0, 1, 2, 3],
                [2, 4, 6, 8]]], dtype=int32)>
        num_heads:2
        d_model:4
        <tf.Tensor: shape=(1, 4, 2, 2), dtype=int32, numpy=
        array([[[[1, 2],
                 [3, 4]],

                [[5, 6],
                 [7, 8]],

                [[0, 1],
                 [2, 3]],

                [[2, 4],
                 [6, 8]]]], dtype=int32)>
        """
        x = tf.reshape(x, (batch_size,-1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0,2,1,3])
    
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        # dimension을 d_model로 동일하게 만들어준다.
        q = self.wq(q) # (batch_size, seq_len, d_model)
        k = self.wk(k) # (batch_size, seq_len, d_model)
        v = self.wv(v) # (batch_size, seq_len, d_model)

        # model_depth -> (num_heads, depth)
        q = self.split_heads(q, batch_size) # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size) # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size) # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        # attention_weights의 shape이 저렇게 나오는 이유는, dot_product를 하기 때문에
        scaled_attention, attention_weights = scaled_dot_product_attention(q,k,v, mask)

        # (num_heads, depth)를 다시 d_model로 변경하기 위해서 차원값을 변경해 줄 필요가 있다.
        scaled_attention = tf.transpose(scaled_attention, perm=[0,2,1,3]) # (batch_size, seq_len_q, num_heads, depth)
        # dense전에 차원값을 d_model로 해야 그 결과가 입력과 같은 값이 된다.
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size,-1,self.d_model)) # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention) # (batch_size, seq_len_q, d_model)

        return output, attention_weights

def MultiHead_test():
    temp_mha = MultiHeadAttention(d_model=512, num_heads=8)
    y = tf.random.uniform((1,60,512))
    out, attn = temp_mha(v=y,k=y,q=y,mask=None)


def scaled_dot_prouct_test():
    def print_out(q,k,v):
        temp_out, temp_attn = scaled_dot_product_attention(
            q, k, v)
        print('Attention weights are:')
        print(temp_attn)
        print('Output is:')
        print(temp_out)

    np.set_printoptions(suppress=True)

    temp_k = tf.constant([[10,0,0],
                          [0,10,0],
                          [0,0,10],
                          [0,0,10]], dtype=tf.float32)  # (4, 3)

    temp_v = tf.constant([[   1,0],
                          [  10,0],
                          [ 100,5],
                          [1000,6]], dtype=tf.float32)  # (4, 2)

    # This `query` aligns with the second `key`,
    # so the second `value` is returned.
    temp_q = tf.constant([[0, 10, 0]], dtype=tf.float32)  # (1, 3)
    print_out(temp_q, temp_k, temp_v)
    temp_q = tf.constant([[0, 0, 10]], dtype=tf.float32)  # (1, 3)
    print_out(temp_q, temp_k, temp_v)
    temp_q = tf.constant([[0, 0, 10], [0, 10, 0], [10, 10, 0]], dtype=tf.float32)  # (3, 3)
    print_out(temp_q, temp_k, temp_v)