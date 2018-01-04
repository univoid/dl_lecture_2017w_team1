import numpy as np
from utils import padding


class Gen_Data_loader():
    def __init__(self, batch_size, seq_length, cond_length, unk):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.cond_length = cond_length
        self.unk = unk
        self.token_stream = []

    def create_batches(self, data_file):
        self.token_stream = []
        with open(data_file, 'r') as f:
            for line in f:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                # unk padding
                self.token_stream.append(padding(parse_line, self.seq_length, self.unk))

        self.num_batch = int(len(self.token_stream) / self.batch_size)
        self.token_stream = self.token_stream[:self.num_batch * self.batch_size]
        self.sequence_batch = np.split(np.array(self.token_stream), self.num_batch, 0)
        self.pointer = 0
        
    def create_cond_batches(self, data_file):
        self.cond_stream = []
        with open(data_file, 'r') as f:
            for line in f:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                self.cond_stream.append(padding(parse_line, self.cond_length, self.unk))
                
        self.num_batch = int(len(self.cond_stream) / self.batch_size)
        self.cond_stream = self.cond_stream[:self.num_batch * self.batch_size]
        self.cond_batch = np.split(np.array(self.cond_stream), self.num_batch, 0)
        self.cond_pointer = 0

    def next_batch(self):
        ret = self.sequence_batch[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret
    
    def next_cond_batch(self):
        ret = self.cond_batch[self.cond_pointer]
        self.cond_pointer = (self.cond_pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0


class Dis_dataloader():
    def __init__(self, batch_size, seq_length, unk):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.unk = unk
        self.sentences = np.array([])
        self.labels = np.array([])

    def load_train_data(self, positive_file, negative_file):
        # Load data
        positive_examples = []
        negative_examples = []
        with open(positive_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                parse_line = padding(parse_line, self.seq_length, self.unk)
                positive_examples.append(parse_line)
        with open(negative_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                parse_line = padding(parse_line, self.seq_length, self.unk)
                negative_examples.append(parse_line)
        num_negative = ((len(positive_examples)+len(negative_examples)) // 
                        self.batch_size) * self.batch_size - len(positive_examples)
        negative_examples = negative_examples[:num_negative]
        self.sentences = np.array(positive_examples + negative_examples)
        # Generate labels
        positive_labels = [[0, 1] for _ in positive_examples]
        negative_labels = [[1, 0] for _ in negative_examples]
        self.labels = np.concatenate([positive_labels, negative_labels], 0)

        # Shuffle the data
        shuffle_indices = np.random.permutation(np.arange(len(self.labels)))
        self.sentences = self.sentences[shuffle_indices]
        self.labels = self.labels[shuffle_indices]

        # Split batches
        self.num_batch = int(len(self.labels) / self.batch_size)
        #self.sentences = self.sentences[:self.num_batch * self.batch_size]
        #self.labels = self.labels[:self.num_batch * self.batch_size]
        self.sentences_batches = np.split(self.sentences, self.num_batch, 0)
        self.labels_batches = np.split(self.labels, self.num_batch, 0)

        self.pointer = 0


    def next_batch(self):
        ret = self.sentences_batches[self.pointer], self.labels_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0

