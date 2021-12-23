import tensorflow as tf
from layers import Encoder, Decoder

class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, target_size, dff, pe_input, pe_target, rate=0.1):
                 # pe: positional encoding
        super(Transformer, self).__init__()

        self.tokenizer = Encoder(num_layers, d_model, num_heads, dff,
                                 pe_input, rate)
        self.decoder = Decoder(num_layers,d_model,num_heads,dff,
                               pe_target, rate)
        
        self.final_layer = tf.keras.layers.Dense(target_size)
    
    def call(self, inp, tar, training, enc_padding_mask,
             look_ahead_mask, dec_padding_mask):
        enc_output = self.tokenizer(inp, training, enc_padding_mask)

        

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask
        )
        final_output = self.final_layer(dec_output)
        final_output = tf.nn.softmax(final_output)
        return final_output, attention_weights
        