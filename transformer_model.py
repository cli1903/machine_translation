import numpy as np
import tensorflow as tf
import transformer_funcs as transformer

class Transformer_Seq2Seq(tf.keras.Model):
    def __init__(self, french_window_size, french_vocab_size, english_window_size, english_vocab_size):

        ######vvv DO NOT CHANGE vvv##################
        super(Transformer_Seq2Seq, self).__init__()

        self.french_vocab_size = french_vocab_size # The size of the french vocab
        self.english_vocab_size = english_vocab_size # The size of the english vocab

        self.french_window_size = french_window_size # The french window size
        self.english_window_size = english_window_size # The english window size
        ######^^^ DO NOT CHANGE ^^^##################


        # TODO:
        # 1) Define any hyperparameters
        # 2) Define embeddings, encoder, decoder, and feed forward layers

        # Define batch size and optimizer/learning rate
        self.batch_size = 100
        self.embedding_size = 64

        # Define english and french embedding layers:
        self.Ef = tf.keras.layers.Embedding(french_vocab_size, self.embedding_size, input_length = self.french_window_size)
        self.Ee = tf.keras.layers.Embedding(english_vocab_size, self.embedding_size, input_length = self.english_window_size)
        
        # Create positional encoder layers
        self.pos_f = transformer.Position_Encoding_Layer(self.french_window_size, self.embedding_size)
        self.pos_e = transformer.Position_Encoding_Layer(self.english_window_size, self.embedding_size)

        # Define encoder and decoder layers:
        self.encoder = transformer.Transformer_Block(self.embedding_size, False)
        
        self.decoder = transformer.Transformer_Block(self.embedding_size, True)
    
        # Define dense layer(s)
        self.dense = tf.keras.layers.Dense(self.english_vocab_size, activation = 'softmax')
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)


    @tf.function
    def call(self, encoder_input, decoder_input):
        """
        :param encoder_input: batched ids corresponding to french sentences
        :param decoder_input: batched ids corresponding to english sentences
        :return prbs: The 3d probabilities as a tensor, [batch_size x window_size x english_vocab_size]
        """
    
        # TODO:
        #1) Add the positional embeddings to french sentence embeddings
        #2) Pass the french sentence embeddings to the encoder
        #3) Add positional embeddings to the english sentence embeddings
        #4) Pass the english embeddings and output of your encoder, to the decoder
        #3) Apply dense layer(s) to the decoder out to generate probabilities
        f_embed = self.Ef(encoder_input)
        pos_embed_f = self.pos_f(f_embed)
        
        encode = self.encoder(pos_embed_f)
        
        e_embed = self.Ee(decoder_input)
        pos_embed_e = self.pos_e(e_embed)
        
        decode = self.decoder(pos_embed_e, encode)
        
        dense = self.dense(decode)
    
        return dense

    def accuracy_function(self, prbs, labels, mask):
        """
        DO NOT CHANGE

        Computes the batch accuracy
        
        :param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
        :param labels:  integer tensor, word prediction labels [batch_size x window_size]
        :param mask:  tensor that acts as a padding mask [batch_size x window_size]
        :return: scalar tensor of accuracy of the batch between 0 and 1
        """

        decoded_symbols = tf.argmax(input=prbs, axis=2)
        accuracy = tf.reduce_mean(tf.boolean_mask(tf.cast(tf.equal(decoded_symbols, labels), dtype=tf.float32),mask))
        return accuracy


    def loss_function(self, prbs, labels, mask):
        """
        Calculates the model cross-entropy loss after one forward pass
        
        :param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
        :param labels:  integer tensor, word prediction labels [batch_size x window_size]
        :param mask:  tensor that acts as a padding mask [batch_size x window_size]
        :return: the loss of the model as a tensor
        """

        # Note: you can reuse this from rnn_model.

        loss = tf.reduce_sum(tf.keras.losses.sparse_categorical_crossentropy(labels, prbs) * mask)
        return loss       

