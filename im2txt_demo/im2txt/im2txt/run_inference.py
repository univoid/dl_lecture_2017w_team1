# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Generate captions for images using default beam search parameters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os


import tensorflow as tf

from im2txt import configuration
from im2txt import inference_wrapper
from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", "",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("vocab_file", "", "Text file containing the vocabulary.")
tf.flags.DEFINE_string("input_files", "",
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")


def main(_):
  # Build the inference graph.
  g = tf.Graph()
  with g.as_default():
    model = inference_wrapper.InferenceWrapper()
    restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                               FLAGS.checkpoint_path)
  g.finalize()

  # Create the vocabulary.
  vocab = vocabulary.Vocabulary(FLAGS.vocab_file)

  filenames = []
  for file_pattern in FLAGS.input_files.split(","):
    filenames.extend(tf.gfile.Glob(file_pattern))
  tf.logging.info("Running caption generation on %d files matching %s",
                  len(filenames), FLAGS.input_files)

  with tf.Session(graph=g) as sess:
    # Load the model from checkpoint.
    restore_fn(sess)

    # Prepare the caption generator. Here we are implicitly using the default
    # beam search parameters. See caption_generator.py for a description of the
    # available beam search parameters.
    generator = caption_generator.CaptionGenerator(model, vocab)

    # prepare translate
    from googletrans import Translator
    translatorGuo = Translator()

    # prepare extraction of keywords
    import urllib2
    import xml.etree.ElementTree as ET
    urlHead = "http://jlp.yahooapis.jp/KeyphraseService/V1/extract?appid=dj00aiZpPXRvZ29acExsMnVWNiZzPWNvbnN1bWVyc2VjcmV0Jng9MDE-&sentence="


    for filename in filenames:
      with tf.gfile.GFile(filename, "r") as f:
        image = f.read()
      captions = generator.beam_search(sess, image)
      print("Captions for image %s:" % os.path.basename(filename))
      keywords = []
      for i, caption in enumerate(captions):
        # Ignore begin and end words.
        sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
        sentence = " ".join(sentence)
        # get japanese sentence
        sentenceJ = translatorGuo.translate(sentence, src='en', dest='ja').text.encode('utf-8')
        print(sentenceJ)
        print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))

        # get keyword
        url = urlHead + sentenceJ
        req = urllib2.Request(url)
        response = urllib2.urlopen(req)
        XmlData = response.read()
        root = ET.fromstring(XmlData)
        for result in root:
            keywords.append(result[0].text.encode('utf-8'))
            
      keywords = set(keywords)
      print("Keywords:")
      for keyword in keywords:
        print(keyword)
      print("Kigo:")
      print(to_kigo(keywords))

      

# from keyword to kigo
def to_kigo(words, top_n=5000):
    import numpy as np
    import pandas as pd
    from gensim.models import word2vec
    model = word2vec.Word2Vec.load('word2vec.gensim.model')
    df = pd.read_csv('kigo.csv', encoding = 'utf8')
    df = df.iloc[:top_n, :]
    V = df.drop(['kigo', 'n'], axis=1).values

    kigo = df.kigo.values

    idx = 0
    max_similarity = 0
    for word in words:
        word = word.decode('utf-8')
        try:
          similarities = model.wv.similarity(kigo, word)
          if similarities.max() >= max_similarity:
            idx = similarities.argmax()
        except Exception:
          pass
        
    
    return(kigo[idx]).encode('utf-8')


if __name__ == "__main__":
  tf.app.run()
