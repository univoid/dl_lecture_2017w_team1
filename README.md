# dl_lecture_2017w_team1

## Requirements: 
* **Tensorflow r1.0.1**
* Python 2.7
* CUDA 7.5+ (For GPU)

## Introduction
> Apply Generative Adversarial Nets to generating sequences of discrete tokens.
> 
> ![](https://github.com/LantaoYu/SeqGAN/blob/master/figures/seqgan.png)
> 
> The illustration of SeqGAN. Left: D is trained over the real data and the generated data by G. Right: G is trained by policy gradient where the final reward signal is provided by D and is passed back to the intermediate action value via Monte Carlo search. 
> 
> The research paper [SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient](http://arxiv.org/abs/1609.05473) has been accepted at the Thirty-First AAAI Conference on Artificial Intelligence (AAAI-17).
> 
（[本家のレポジトリ](https://github.com/LantaoYu/SeqGAN)より引用）

## How to use
### 1.Twitterからデータ収集

```
$ python timeline.py
```
でNON STYLE 石田の「おはようございます。みなさんの」から始まるツイートを全て取ってこれます。

出力先は```save/raw_tweet.py```

他のアカウントからツイートを持ってきたければ、24行目と48行目の"screen_name"を任意のアカウント名の@以下に変えて、

39〜41行目を削除して使ってください。

### 2.日本語のトークナイズ

```
$ python datacleaner.py
```

```save/raw_tweet.py```からURLを除去後、形態素解析をMeCabで行います。

出力先は```save/parsed_tweet.py```

### 3.SeqGANの学習

```
$ python sequence_gan.py
```
