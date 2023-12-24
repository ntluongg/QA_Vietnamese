import subprocess
import sys
import torch
import os
import cv2
import numpy as np
import pandas as pd
import argparse
import torch.nn as nn
from transformers import BertTokenizer, BertConfig
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from QA_Vietnamese.train.model.bert.BERT_model import QaBERT
from QA_Vietnamese.train.model.bert.train_bert import evaluate
from QA_Vietnamese.train.model.bert.dataloader import load_squad_to_torch_dataset

class Args:
    do_lower_case = True
    folder_model = ""

    path_input_test_data = "/kaggle/working/QA_Vietnamese/train/dataset/data/test_data.csv"

    no_cuda = False
    n_gpu = 1
    device = "cuda:0"
    seed = 42

    max_seq_length = 400
    max_query_length = 64
    weight_class = [1, 1]


args = Args()

device = torch.device(args.device)
tokenizer = BertTokenizer.from_pretrained(args.folder_model, do_lower_case=args.do_lower_case)

config = BertConfig.from_pretrained(args.folder_model)

# # custom some parameter for custom bert
config = config.to_dict()
config.update({"device": args.device})
config = BertConfig.from_dict(config)

model = QaBERT.from_pretrained(args.folder_model, config=config)

model = model.to(device)

# Load the checkpoint
parser = argparse.ArgumentParser(description='Bert Inference')
parser.add_argument('--checkpoint', type=str, help='Path to the model checkpoint')
parser.add_argument('--test_dir', type=str, help='Directory path to test')
A = parser.parse_args()


path_input_data = args.path_input_test_data
#change to your desired folder
path_input_data_pt = "/kaggle/working/test_data.pt"


eval_dataset, eval_dataloader = load_squad_to_torch_dataset(path_input_data,
                                                            tokenizer,
                                                            args.max_seq_length,
                                                            args.max_query_length,
                                                            12,
                                                            is_training=True)
torch.save(eval_dataset, path_input_data_pt)
    
# Eval!
print("***** Running evaluation")
print("  Num examples = %d", len(eval_dataset))

total_loss = 0.0
l_full_predict = []
l_full_target = []

for batch in tqdm(eval_dataloader, desc="Evaluating", leave=False):
    model.eval()
    batch = tuple(t.to(args.device) for t in batch)
    with torch.no_grad():
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'token_type_ids': batch[2],
                  'label': batch[3]
                  }

        loss, l_predict, l_target = model.compute_loss(inputs['input_ids'],
                                inputs['attention_mask'],
                                inputs['token_type_ids'],
                                inputs['label'])
        total_loss += loss.item()
        l_full_predict.extend(l_predict)
        l_full_target.extend(l_target)

path_input_data = args.path_input_test_data
#change to your desired folder
path_input_data_pt = "/kaggle/working/test_final.pt"

l_old_eval = len(eval_dataset)

eval_dataset, eval_dataloader = load_squad_to_torch_dataset("/kaggle/working/QA_Vietnamese/train/dataset/data/test_final.csv",
                                                            tokenizer,
                                                            args.max_seq_length,
                                                            args.max_query_length,
                                                            12,
                                                            is_training=True)
torch.save(eval_dataset, path_input_data_pt)
        
for batch in tqdm(eval_dataloader, desc="Evaluating", leave=False):
    model.eval()
    batch = tuple(t.to(args.device) for t in batch)
    with torch.no_grad():
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'token_type_ids': batch[2],
                  'label': batch[3]
                  }

        loss, l_predict, l_target = model.compute_loss(inputs['input_ids'],
                                inputs['attention_mask'],
                                inputs['token_type_ids'],
                                inputs['label'])
        total_loss += loss.item()
        l_full_predict.extend(l_predict)
        l_full_target.extend(l_target)
        
f1_score_micro = f1_score(l_full_target, l_full_predict)
accuracy = accuracy_score(l_full_target, l_full_predict)
precision = precision_score(l_full_target, l_full_predict)
recall = recall_score(l_full_target, l_full_predict)

output_validation = {
    "loss": round(total_loss / (len(eval_dataset)+ l_old_eval), 3),
    "accuracy": round(accuracy, 3),
    "f1": round(f1_score_micro, 3),
    "precision": round(precision, 3),
    "recall": round(recall, 3)
}

print(output_validation)