import pickle
import tensorflow as tf
from generator import Generator
from discriminator import Discriminator

#########################################################################################
#  Generator  Hyper-parameters
######################################################################################
EMB_DIM = 32 # embedding dimension
HIDDEN_DIM = 32 # hidden state dimension of lstm cell
SEQ_LENGTH = 17 # sequence length
COND_LENGTH = 7 # condition length
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


def main():
    with open('save/token2id.pickle') as f:
        token2id = pickle.load(f)
    vocab_size=len(token2id)
    
    generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, COND_LENGTH, START_TOKEN, is_cond=1)
    discriminator = Discriminator(sequence_length=SEQ_LENGTH, num_classes=2, vocab_size=vocab_size, embedding_size=dis_embedding_dim, 
                                filter_sizes=dis_filter_sizes, num_filters=dis_num_filters, l2_reg_lambda=dis_l2_reg_lambda)
    
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, 'save/haiku_generator')
    
    
    
if __name__ == '__main__':
    main()