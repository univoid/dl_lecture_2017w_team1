import argparse
import numpy as np
import tensorflow as tf
import random
from dataloader import Gen_Data_loader, Dis_dataloader
from generator import Generator
from discriminator import Discriminator
from rollout import ROLLOUT
from target_lstm import TARGET_LSTM
from vocabulary import Vocab
import cPickle
import pickle
from clock import Clock

#########################################################################################
#  Generator  Hyper-parameters
######################################################################################
EMB_DIM = 32 # embedding dimension
HIDDEN_DIM = 32 # hidden state dimension of lstm cell
SEQ_LENGTH = 17 # sequence length
COND_LENGTH = 7 # condition length
START_TOKEN = 0
PRE_EPOCH_GEN_NUM = 1 # supervise (maximum likelihood estimation) epochs
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
PRE_EPOCH_DIS_NUM = 1

#########################################################################################
#  Basic Training Parameters
#########################################################################################
TOTAL_BATCH = 1
parsed_tweet_file = 'save/parsed_tweet.txt'
parsed_haiku_file = 'save/kanji_haiku.txt'
parsed_kigo_file = 'save/kanji_kigo.txt'
generated_tweet_file = 'save/generated_tweet_{}.txt'
generated_haiku_file = 'save/kanji_generated_haiku_{}.txt'
generated_haiku_with_kigo_file = 'save/kanji_generated_haiku_with_kigo_{}.txt'
positive_file = 'save/real_data.txt'
positive_condition_file = 'save/real_condition_data.txt'
negative_file = 'save/generator_sample.txt'
eval_file = 'save/eval_file.txt'
generated_num = 10000


def generate_samples(sess, trainable_model, batch_size, generated_num, output_file, cond, vocab):
    # Generate Samples
    generated_samples = []
    if cond:
        cond_samples = []
        for _ in range(int(generated_num / batch_size)):
            cond_batch = vocab.choice_cond(batch_size)
            cond_samples.extend(cond_batch)
            generated_samples.extend(trainable_model.generate(sess, cond=cond_batch))
    else:
        for _ in range(int(generated_num / batch_size)):
            generated_samples.extend(trainable_model.generate(sess))

    with open(output_file, 'w') as fout:
        if cond:
            for c, poem in zip(cond_samples, generated_samples):
                header = ' '.join([str(x) for x in c]+[str(vocab.dic.token2id[u','])]) + ' '
                buffer = ' '.join([str(x) for x in poem]) + '\n'
                fout.write(header + buffer)
        else:
            for poem in generated_samples:
                buffer = ' '.join([str(x) for x in poem]) + '\n'
                fout.write(buffer)


def target_loss(sess, target_lstm, data_loader):
    # target_loss means the oracle negative log-likelihood tested with the oracle model "target_lstm"
    # For more details, please see the Section 4 in https://arxiv.org/abs/1609.05473
    nll = []
    data_loader.reset_pointer()

    for it in xrange(data_loader.num_batch):
        batch = data_loader.next_batch()
        g_loss = sess.run(target_lstm.pretrain_loss, {target_lstm.x: batch})
        nll.append(g_loss)

    return np.mean(nll)


def pre_train_epoch(sess, trainable_model, data_loader, cond=0):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in xrange(data_loader.num_batch):
        batch = data_loader.next_batch()
        if cond:
            cond_batch = data_loader.next_cond_batch()
            _, g_loss = trainable_model.pretrain_step(sess, batch, cond=cond_batch)
        else:
            _, g_loss = trainable_model.pretrain_step(sess, batch)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)


def main():
    clock = Clock()
    clock.start()
    random.seed(SEED)
    np.random.seed(SEED)
    assert START_TOKEN == 0
    
    parser = argparse.ArgumentParser(description='conditional SeqGAN')
    parser.add_argument('--conditional', '-c', type=int, default=0,
                        help='If you make SeqGAN conditional, set `-c` 1.')
    args = parser.parse_args()
    cond = args.conditional
    
    vocab = Vocab()
    vocab.construct(parsed_haiku_file)
    vocab.word2id(parsed_haiku_file, positive_file)
    UNK = vocab.dic.token2id[u'<UNK>']
    COMMA = vocab.dic.token2id[u',']

    gen_data_loader = Gen_Data_loader(BATCH_SIZE, SEQ_LENGTH, COND_LENGTH, UNK)
    # likelihood_data_loader = Gen_Data_loader(BATCH_SIZE, SEQ_LENGTH, COND_LENGTH, UNK) # For testing
    vocab_size = len(vocab.dic.token2id)
    with open('save/token2id.pickle', 'w') as f:
        pickle.dump(vocab.dic.token2id, f)
    dis_data_loader = Dis_dataloader(BATCH_SIZE, SEQ_LENGTH, UNK)

    generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, COND_LENGTH, START_TOKEN, is_cond=cond)
    # target_params = cPickle.load(open('save/target_params.pkl'))
    # target_lstm = TARGET_LSTM(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN, target_params) # The oracle model

    discriminator = Discriminator(sequence_length=SEQ_LENGTH, num_classes=2, vocab_size=vocab_size, embedding_size=dis_embedding_dim, 
                                filter_sizes=dis_filter_sizes, num_filters=dis_num_filters, l2_reg_lambda=dis_l2_reg_lambda)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # First, use the oracle model to provide the positive examples, which are sampled from the oracle data distribution
    # generate_samples(sess, target_lstm, BATCH_SIZE, generated_num, positive_file)
    gen_data_loader.create_batches(positive_file)

    if cond:
        vocab.word2id(parsed_kigo_file, positive_condition_file)
        vocab.load_cond(positive_condition_file, COND_LENGTH, UNK)
        gen_data_loader.create_cond_batches(positive_condition_file)

    log = open('save/experiment-log.txt', 'w')
    #  pre-train generator
    print 'Start pre-training...'
    log.write('pre-training...\n')
    for epoch in xrange(PRE_EPOCH_GEN_NUM):
        loss = pre_train_epoch(sess, generator, gen_data_loader, cond=cond)
        if epoch % 5 == 0:
            generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file, cond, vocab)
            # likelihood_data_loader.create_batches(eval_file)
            # test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
            # print 'pre-train epoch ', epoch, 'test_loss ', test_loss
            # buffer = 'epoch:\t'+ str(epoch) + '\tnll:\t' + str(test_loss) + '\n'
            # log.write(buffer)
    clock.check_HMS()
    
    print 'Start pre-training discriminator...'
    # Train 3 epoch on the generated data and do this for 50 times
    for _ in range(PRE_EPOCH_DIS_NUM):
        generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file, cond, vocab)
        dis_data_loader.load_train_data(positive_file, negative_file)
        for _ in range(3):
            dis_data_loader.reset_pointer()
            for it in xrange(dis_data_loader.num_batch):
                x_batch, y_batch = dis_data_loader.next_batch()
                feed = {
                    discriminator.input_x: x_batch,
                    discriminator.input_y: y_batch,
                    discriminator.dropout_keep_prob: dis_dropout_keep_prob
                }
                _ = sess.run(discriminator.train_op, feed)
    clock.check_HMS()

    rollout = ROLLOUT(generator, 0.8, SEQ_LENGTH)

    print '#########################################################################'
    print 'Start Adversarial Training...'
    log.write('adversarial training...\n')
    for total_batch in range(TOTAL_BATCH):
        # Train the generator for one step
        for it in range(1):
            if cond:
                cond_batch = vocab.choice_cond(BATCH_SIZE)
                samples = generator.generate(sess, cond=cond_batch)
            else:
                samples = generator.generate(sess)
            rewards = rollout.get_reward(sess, samples, 16, discriminator)
            feed = {generator.x: samples, generator.rewards: rewards}
            if cond:
                feed[generator.cond] = cond_batch
            _ = sess.run(generator.g_updates, feed_dict=feed)

        # Test
        if total_batch % 5 == 0 or total_batch == TOTAL_BATCH - 1:
            generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file, cond, vocab)
            # likelihood_data_loader.create_batches(eval_file)
            # test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
            # buffer = 'epoch:\t' + str(total_batch) + '\tnll:\t' + str(test_loss) + '\n'
            # print 'total_batch: ', total_batch, 'test_loss: ', test_loss
            # log.write(buffer)
            if cond:
                vocab.id2word(eval_file, generated_haiku_with_kigo_file.format(total_batch))
            else:
                vocab.id2word(eval_file, generated_haiku_file.format(total_batch))

        # Update roll-out parameters
        rollout.update_params()

        # Train the discriminator
        for _ in range(5):
            generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file, cond, vocab)
            dis_data_loader.load_train_data(positive_file, negative_file)

            for _ in range(3):
                dis_data_loader.reset_pointer()
                for it in xrange(dis_data_loader.num_batch):
                    x_batch, y_batch = dis_data_loader.next_batch()
                    feed = {
                        discriminator.input_x: x_batch,
                        discriminator.input_y: y_batch,
                        discriminator.dropout_keep_prob: dis_dropout_keep_prob
                    }
                    _ = sess.run(discriminator.train_op, feed)
    clock.check_HMS()
    saver = tf.train.Saver()
    saver.save(sess, "save/haiku_generator")
    log.close()


if __name__ == '__main__':
    main()
