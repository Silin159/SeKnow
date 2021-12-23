# SeKnow-S2S
This is the code for SeKnow-S2S in paper [End-to-End Task-Oriented Dialog Modeling with Semi-Structured Knowledge Management](https://arxiv.org/abs/2106.11796).

## Requirements
- Python 3.6
- PyTorch 1.2.0
- NLTK 3.4.5
- Spacy 2.2.2

We use some NLP tools in NLTK which can be installed through:
```
python -m nltk.downloader stopwords punkt wordnet
```

## Dataset
1. Raw dataset: [modified MultiWOZ 2.1](https://github.com/alexa/alexa-with-dstc9-track1-dataset)

2. Our preprocessed dataset can be downloaded from [this link](https://drive.google.com/file/d/1THzk_MXPeSp3wWBk19sdCh7l-oDhU4Fq/view?usp=sharing), please unzip the file under the root directory and data is placed in data/.

## Implementations of SeKnow-S2S
We build SeKnow-S2S in both single-decoder and multi-decoder belief state decoding implementations.

   1.SeKnow-S2S with single-decoder belief state decoding implementation: ``Single/``
   
   2.SeKnow-S2S with multi-decoder belief state decoding implementation: ``Multiple/``

## Running Experiments
Before running, place the preprocessed dataset ``data/`` into ``Single/`` or ``Multiple/``.
Go to the experiment root:
```
cd Single
```
or
```
cd Multiple
```

### Training
```
python train.py -mode train -dataset multiwoz -method bssmc -c spv_proportion=100 exp_no=your_exp_name
```

### Testing
```
python train.py -mode test -dataset multiwoz -method bssmc -c eval_load_path=[experimental path]
```
