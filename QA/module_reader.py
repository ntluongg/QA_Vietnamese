from __future__ import absolute_import, division, print_function

from transformers import AutoTokenizer, BertForQuestionAnswering
from multiprocessing import Process, Pool
import torch
import logging
import sys
import torch.nn.functional as F

class Args:
    bert_model = './resources'
    max_seq_length = 160
    doc_stride = 160
    predict_batch_size = 20
    n_best_size=20
    max_answer_length=30
    verbose_logging = False
    no_cuda = True
    seed= 42
    do_lower_case= True
    version_2_with_negative = True
    null_score_diff_threshold=0.0
    max_query_length = 64
    THRESH_HOLD = 0.95
    
args=Args()


class Reader():
    def __init__(self, model):
        self.log = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("/kaggle/working/final_bert", do_lower_case= True)
        self.model = model
        self.model.to(self.device)
        self.model.eval()
        self.args = args
    
    def getPredictions(self,questions, contexts):
        question   = question.replace('_',' ')
        paragraphs = [p.replace('_',' ') for p in paragraphs]
        
        # Ensure that the number of contexts and questions are the same
        assert len(contexts) == len(questions)

        # Loop over the list of contexts and questions
        for context, question in zip(contexts, questions):
            # Encode the context and question
            inputs = self.tokenizer.encode_plus(question, context, return_tensors='pt')

            # Get the model's predictions
            answer_start_scores, answer_end_scores = self.model(**inputs)

            # Apply softmax function to convert logits to probabilities
            start_probs = F.softmax(answer_start_scores, dim=-1)
            end_probs = F.softmax(answer_end_scores, dim=-1)

            # Get the start and end positions with the highest probability
            answer_start = torch.argmax(start_probs)
            answer_end = torch.argmax(end_probs) + 1

            # Get the confidence scores for the start and end positions
            start_confidence = start_probs[0][answer_start].item()
            end_confidence = end_probs[0][answer_end].item()

            # Print the start and end positions and their confidence scores
            print(f"Question: {question}")
            print(f"Context: {context}")
            print(f"Start position: {answer_start}, confidence score: {start_confidence}")
            print(f"End position: {answer_end}, confidence score: {end_confidence}\n")
            print(f"Answer: {context[answer_start: answer_end+1]}")