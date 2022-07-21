import os
import numpy as np
import tensorflow as tf
import numpy as np
from preprocess import *
from transformer_model import Transformer_Seq2Seq
from rnn_model import RNN_Seq2Seq
import sys


def train(model, train_french, train_english, eng_padding_index):
    """
    Runs through one epoch - all training examples.

    :param model: the initilized model to use for forward and backward pass
    :param train_french: french train data (all data for training) of shape (num_sentences, 14)
    :param train_english: english train data (all data for training) of shape (num_sentences, 15)
    :param eng_padding_index: the padding index, the id of *PAD* token. This integer is used to mask padding labels.
    :return: None
    """

    # NOTE: For each training step, you should pass in the french sentences to be used by the encoder, 
    # and english sentences to be used by the decoder
    # - The english sentences passed to the decoder have the last token in the window removed:
    #     [STOP CS147 is the best class. STOP *PAD*] --> [STOP CS147 is the best class. STOP] 
    # 
    # - When computing loss, the decoder labels should have the first word removed:
    #     [STOP CS147 is the best class. STOP] --> [CS147 is the best class. STOP] 
    
    

    for i in range(0, len(train_french), model.batch_size):
        encode_inputs = train_french[i : i + model.batch_size]
        decode_inputs = np.array([x[:-1] for x in train_english[i : i + model.batch_size]], dtype = np.float32)
        labels = np.array([x[1:] for x in train_english[i : i + model.batch_size]], dtype = np.float32)
        
        with tf.GradientTape() as tape:
            prob = model.call(encode_inputs, decode_inputs)
            
            bool_mask = tf.cast(labels != eng_padding_index, tf.float32)
            
            loss = model.loss_function(prob, labels, bool_mask) / tf.cast(tf.size(bool_mask), tf.float32)
            
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))



def test(model, test_french, test_english, eng_padding_index):
    """
    Runs through one epoch - all testing examples.

    :param model: the initilized model to use for forward and backward pass
    :param test_french: french test data (all data for testing) of shape (num_sentences, 14)
    :param test_english: english test data (all data for testing) of shape (num_sentences, 15)
    :param eng_padding_index: the padding index, the id of *PAD* token. This integer is used to mask padding labels.
    :returns: perplexity of the test set, per symbol accuracy on test set
    """

    # Note: Follow the same procedure as in train() to construct batches of data!
    
    total_loss = 0.0
    total_acc = 0.0
    tot_tok = 0.0
    

    for i in range(0, len(test_french), model.batch_size):
        encode_inputs = test_french[i : i + model.batch_size]
        decode_inputs = np.array([x[:-1] for x in test_english[i : i + model.batch_size]], dtype = np.float32)
        labels = np.array([x[1:] for x in test_english[i : i + model.batch_size]], dtype = np.float32)
        
        prob = model.call(encode_inputs, decode_inputs)
        
        bool_mask = tf.cast(labels != eng_padding_index, tf.float32)
        
        num_tok = tf.reduce_sum(bool_mask)
        tot_tok += num_tok
    
        total_acc += model.accuracy_function(prob, labels, bool_mask) * num_tok
        
        #print(model.loss_function(prob, labels, bool_mask) / tot_size)
        
        total_loss += model.loss_function(prob, labels, bool_mask)
        
        
    perplexity = np.exp(total_loss/tot_tok)
    acc = total_acc / tot_tok
    return perplexity, acc

def main():    
    if len(sys.argv) != 2 or sys.argv[1] not in {"RNN","TRANSFORMER"}:
            print("USAGE: python assignment.py <Model Type>")
            print("<Model Type>: [RNN/TRANSFORMER]")
            exit()

    print("Running preprocessing...")
    train_english,test_english, train_french,test_french, english_vocab,french_vocab,eng_padding_index = get_data('data/fls.txt','data/els.txt','data/flt.txt','data/elt.txt')
    print("Preprocessing complete.")

    model_args = (FRENCH_WINDOW_SIZE,len(french_vocab),ENGLISH_WINDOW_SIZE, len(english_vocab))
    if sys.argv[1] == "RNN":
        model = RNN_Seq2Seq(*model_args)
    elif sys.argv[1] == "TRANSFORMER":
        model = Transformer_Seq2Seq(*model_args) 
    
    
    # TODO:
    # Train and Test Model for 1 epoch.
    print("Training")
    train(model, train_french, train_english, eng_padding_index)
    print("Testing")
    perp, acc = test(model, test_french, test_english, eng_padding_index)
    print("Perplexity: ")
    print(perp)
    print("Accuracy: ")
    print(acc)

if __name__ == '__main__':
   main()


