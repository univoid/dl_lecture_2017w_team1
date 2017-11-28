from gensim.corpora.dictionary import Dictionary
import codecs

class Vocab():
    def __init__(self):
        self.dic = Dictionary()
        self.dic.add_documents([[u'<UNK>']])
        
    def construct(self, input_file):
        f = codecs.open(input_file, 'r', 'utf-8')
        sentences = []
        for line in f:
            line = line.strip().split()
            sentences.append(line)
        self.dic.add_documents(sentences)
        f.close()
    
    def word2id(self, input_file, output_file):
        f = codecs.open(input_file, 'r', 'utf-8')
        g = open(output_file, 'w')
        for line in f:
            line = line.strip().split()
            line = map(lambda x: str(self.dic.token2id[x]), line)
            line = u" ".join(line) + u"\n"
            g.write(line)
        f.close()
        g.close()
        
    def id2word(self, input_file, output_file):
        f = open(input_file, 'r')
        g = codecs.open(output_file, 'w', 'utf-8')
        for line in f:
            line = line.strip().split()
            line = map(lambda x: self.dic.id2token[int(x)], line)
            line = u" ".join(line) + u"\n"
            g.write(line)
        f.close()
        g.close()