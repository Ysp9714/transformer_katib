import tensorflow as tf
from attention import MultiHeadAttention
from utils import positional_encoding


def point_wise_feed_forward_network(d_model, dff):
    # 논문에서 첫번째 덴스에서 4배로 Dimension을 증가시킨다. 
    # 그 값이 dff 이다.
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='gelu'),
        tf.keras.layers.Dense(d_model)
    ])

# Encoder
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model,num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
    
    def call(self, x, training, mask):
        attn_output, _ = self.mha(x,x,x,mask) # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output) # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1) # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output) # (batch_size, input_seq_len, d_model)

        return out2

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)


    def call(self,x, enc_output, training, look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask) # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training) 
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask) # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1) # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2

''' Encoder
Encoder는 다음으로 구성된다.
1. 입력 임베딩
2. 위치 인코딩
3. N 인코더 레이어
'''
class Encoder(tf.keras.layers.Layer):
    # dff: depth of feed forward
    def __init__(self, num_layers, d_model, num_heads, dff,
                 maximum_position_encoding, rate=0.1):
        super(Encoder,self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Dense(self.d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)
        self.pos_emb = tf.keras.layers.Dense(self.d_model)
        self.enc_layers = [EncoderLayer(self.d_model, num_heads, dff, rate) for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self,x, training, mask):
        seq_len = tf.shape(x)[1] # seq_len
        
        # adding embedding and position encoding.
        x = self.embedding(x) # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model,tf.float32)) # scale
        # 내가 필요한 값은 이번에 인풋으로 들어온 값의 seq_len까지의 값이기 때문에 seq_len에서 끊는다.
        # x += self.pos_encoding[:, :seq_len, :]
        x += self.pos_emb(self.pos_encoding[:, :seq_len, :])
        x = self.dropout(x, training=training)

        for enc in self.enc_layers:
            x = enc(x, training, mask)

        return x # (batch_size, input_seq_len, d_model)

''' Decoder
Decoder는 다음으로 구성된다.
1. 출력 임베딩
2. 위치 인코딩
3. N 디코더 레이어
'''

class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Dense(d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.decoder_layers = [DecoderLayer(d_model, num_heads, dff, rate) 
                                for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)
    
    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)
        # x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        # x += self.pos_encoding[:,:seq_len]
        # x = self.dropout(x,training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.decoder_layers[i](x, enc_output,training,
                                                       look_ahead_mask, padding_mask)
            attention_weights[f'decoder_layer{i+1}_block1'] = block1
            attention_weights[f'decoder_layer{i+1}_block2'] = block2
        
        return x, attention_weights
