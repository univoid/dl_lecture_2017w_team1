from gensim.corpora.dictionary import Dictionary
import codecs
import numpy as np
from utils import padding


class Vocab():
    def __init__(self):
        self.dic = Dictionary()
        self.dic.add_documents([[u'<UNK>', u',']])
        
    def construct(self, input_file):
        f = codecs.open(input_file, 'r', 'utf-8')
        sentences = []
        for line in f:
            line = line.strip().split()
            sentences.append(line)
        self.dic.add_documents(sentences)
        f.close()
        self.dic.id2token = {v:k for k, v in self.dic.token2id.items()}
        
    def load_cond(self, input_file, cond_length, unk):
        """Get a list of unique conditions"""
        f = codecs.open(input_file, 'r', 'utf-8')
        conditions = []
        lines = f.readlines()
        if lines[-1].strip()=='':
            print("deleted the last element:", lines[-1])
            lines=lines[:-1]
        lines = list(set(lines))
        for line in lines:
            line = line.strip().split()
            line = padding(line, cond_length, unk)
            if not line in conditions:
                conditions.append(line)
        self.cond = np.array(conditions)
        self.n_cond = len(conditions)
        
    def choice_cond(self, num):
        return self.cond[np.random.choice(len(self.cond), num)]
    
    def word2id(self, input_file, output_file):
        def get_id(dic, key):
            if key in dic:
                return str(dic[key])
            else:
                ret=[]
                key=list(key)
                for k in key:
                    ret.append(str(dic.get(k, 0)))
                return u" ".join(ret)
            
        f = codecs.open(input_file, 'r', 'utf-8')
        g = open(output_file, 'w')
        for line in f:
            line = line.strip().split()
            line = map(lambda x: get_id(self.dic.token2id, x), line)
            line = u" ".join(line) + u"\n"
            g.write(line)
        f.close()
        g.close()
        
    def id2word(self, input_file, output_file):
        f = open(input_file, 'r')
        g = codecs.open(output_file, 'w', 'utf-8')
        for line in f:
            line = line.strip().split()
            line = map(lambda x: self.dic.id2token.get(int(x), u'#'), line)
            line = u" ".join(line) + u"\n"
            g.write(line)
        f.close()
        g.close()