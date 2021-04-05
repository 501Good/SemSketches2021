# SemSketches2021
The solution for the Task 1 of the SemSketches competition for the Dialog 2021. (https://github.com/dialogue-evaluation/SemSketches)

# Results
| dataset  | score |
|----------|-------|
|dev       |0.104  |
|manual dev|0.127  |

# Requirements

## Python
- [sentence_transformers](https://github.com/UKPLab/sentence-transformers)
- torch
- numpy
- tqdm

# Instructions

## Training
1. Clone the [SemSketches repository](https://github.com/dialogue-evaluation/SemSketches) and copy the `data` folder;
2. Download the [PyTorch Sentence RuBERT, Russian, cased by DeepPavlov](http://docs.deeppavlov.ai/en/master/features/models/bert.html) and unpack the `pytorch_model.bin` and `vocab.txt` files into `sentence_ru_cased_pipe_L-12_H-768_A-12_pt/0_Transformer/` folder;
3. Follow the steps in the `SentenceTransformersSiamese.ipynb`.

## Inference
1. [Download the trained model](https://drive.google.com/file/d/1a3iViYBPmQy9xSgcIKRCfrJOPUNQrgju/view?usp=sharing) and unpack it into the main directory;
2. Follow the steps in the `SentenceTransformersSiamese.ipynb`.