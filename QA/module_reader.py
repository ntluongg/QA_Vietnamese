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
        contexts = [p.replace('_',' ') for p in contexts]
        

        # Initialize a list to store the confidence scores and answers
        confidence_scores_and_answers = []

        # Loop over the list of contexts
        for context in contexts:
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

            # Decode the answer
            answer = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))

            # Store the confidence score and answer in the list
            confidence_scores_and_answers.append((start_confidence * end_confidence, answer))

        # Sort the list in descending order of confidence score
        confidence_scores_and_answers.sort(key=lambda x: x[0], reverse=True)

        # Get the top 5 confidence scores and answers
        top_5_confidence_scores_and_answers = confidence_scores_and_answers[:5]

        # Print the top 5 confidence scores and answers
        for score, answer in top_5_confidence_scores_and_answers:
            print(f"Answer: {answer}, Confidence score: {score}\n")