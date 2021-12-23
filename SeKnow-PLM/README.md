# SeKnow-PLM
This is the code for SeKnow-PLM in paper [End-to-End Task-Oriented Dialog Modeling with Semi-Structured Knowledge Management](https://arxiv.org/abs/2106.11796).

## Getting started
Start with creating a python 3.7 venv and installing requirements.txt.

## Datasets
Our preprocessed datasets can be downloaded from [this link](https://drive.google.com/file/d/1laIY5xufen5RYpc8_Jtw9FaG8dxyKSDw/view?usp=sharing), please unzip the file under the SeKnow-PLM root directory and data is placed in datasets/.

Please also unzip the back translation corpus for data augmentation in scripts/data/backtranslation.zip

## Training and Evaluation
Go to the experiment root:
```
cd scripts
```
### Pre-Training (Skip if directly using AuGPT)
We directly use AuGPT， which is GPT-2 further pre-trained on Taskmaster-1 and Schema Guided Dialogue.
AuGPT can be downloaded from the Hugging Face model repository as `jkulhanek/augpt-bigdata`.
If you want to further pre-train GPT-2 by yourself:
```bash
train.py --epochs 8 --restrict-domains --train-dataset schemaguided-train+taskmaster-train --dev-dataset schemaguided-dev+taskmaster-dev --validation-steps 10000 --logging-steps 1000 --warmup-steps 5000 --evaluation-dialogs 0 --fp16
```

### Fine-Tuning and Evaluation
```
bash train_multiwoz.sh
```
