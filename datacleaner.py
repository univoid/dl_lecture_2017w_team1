# coding: utf-8
import codecs
import re
import MeCab

dic_path="/usr/local/lib/mecab/dic/mecab-ipadic-neologd"
tagger = MeCab.Tagger("-Ochasen -d {0}".format(dic_path))

def _tokenize(text):
    text = text.encode('utf-8')
    sentence = []
    node = tagger.parse(text)
    node = node.split("\n")
    for i in range(len(node)):
        feature = node[i].split("\t")
        if feature[0] == "EOS":
            break
        sentence.append(feature[0].decode('utf-8'))
    return sentence

def main():
    f = codecs.open("./save/raw_data.txt", "r", "utf-8")
    g = codecs.open("./save/real_data.txt", "w", "utf-8")
    for line in f:
        line = re.sub(r"http.*","",line)
        line.strip()
        sentence = _tokenize(line)
        g.write(u" ".join(sentence))
        if sentence[-1] != u"。":
            g.write(u" 。")
        g.write(u"\n")
    f.close()
    g.close()

if __name__ == "__main__":
    main()
