# coding: utf-8

import re
import pickle
import numpy as np
import tensorflow as tf
from generator import Generator
from discriminator import Discriminator
from utils import padding

#########################################################################################
#  Generator  Hyper-parameters
######################################################################################
EMB_DIM = 32 # embedding dimension
HIDDEN_DIM = 32 # hidden state dimension of lstm cell
SEQ_LENGTH = 17 # sequence length
COND_LENGTH = 1 # condition length
START_TOKEN = 0
SEED = 88
BATCH_SIZE = 16

#########################################################################################
#  Discriminator  Hyper-parameters
#########################################################################################
dis_embedding_dim = 64
dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7] # each element should be less than SEQ_LENGTH
dis_num_filters = [100, 200, 200, 200, 200, 100, 100]
dis_dropout_keep_prob = 0.75
dis_l2_reg_lambda = 0.2
dis_batch_size = 16

#########################################################################################
#  Other Hyper-parameters
#########################################################################################
path_to_token2id = 'save/conddis_token2id.pickle'
path_to_generator = 'save/conddis_haiku_generator'


def get_haiku_conddis(kigo):
    with open(path_to_token2id) as f:
        token2id = pickle.load(f)
    vocab_size=len(token2id)
    UNK = token2id.get('<UNK>', 0)
    
    generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, COND_LENGTH, START_TOKEN, is_cond=1)
    discriminator = Discriminator(sequence_length=SEQ_LENGTH, cond_length=COND_LENGTH, num_classes=2, vocab_size=vocab_size, embedding_size=dis_embedding_dim, 
                                filter_sizes=dis_filter_sizes, num_filters=dis_num_filters, l2_reg_lambda=dis_l2_reg_lambda, is_cond=1)
    
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, path_to_generator)
    
    if isinstance(kigo, str):
        kigo = kigo.decode('utf-8')
    cond = map(lambda x: token2id.get(x, 0), [kigo])
    cond = np.array(padding(cond, COND_LENGTH, UNK)*BATCH_SIZE).reshape(BATCH_SIZE, COND_LENGTH)
    generated_sequences = generator.generate(sess, cond=cond)
    
    id2token = {k:v for v,k in token2id.items()}
    generated_haikus = map(lambda y: map(lambda x: id2token.get(x, '<UNK>'), y), generated_sequences)
    
    generated_haikus = map(lambda x: re.sub(r' <UNK>', '',' '.join(x)), generated_haikus)
    
    return generated_haikus