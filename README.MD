# Multi Class Sentence Classification using SciBERT 


This repository contains code for the code for multi class text classification using PUBMED 200k RCT data.
It utilised following artifacts for text classification.

  - Pretreined Scibert model (scibert_scivocab_uncased)
  - Training data : Pubmed 200k RCT
  - implemented using pytorch , huggingface
  - achieves ~90% accuracy 

### How to run ?
Install the dependencies mentioned in requirements.txt.

```sh

$pip install -r requirements.txt
$./download_scibert_embeddings.sh
$ cd src
$ python3 sentence_classifier.py --mode train 
$ python3 sentence_classifier.py --mode eval 
```


### Referenes



|References  | Links |
| ------ | ------ |
|BERT  |  https://arxiv.org/abs/1810.04805|
| SciBERT | https://github.com/allenai/scibert |
|huggingface | https://huggingface.co/transformers/ |
| Muti class classification blog & code reference | ttps://medium.com/@aniruddha.choudhury94/part-2-bert-fine-tuning-tutorial-with-pytorch-for-text-classification-on-the-corpus-of-linguistic-18057ce330e1 |
|PUBMED 200k RCT|https://github.com/Franck-Dernoncourt/pubmed-rct/tree/master/PubMed_200k_RCT.|






