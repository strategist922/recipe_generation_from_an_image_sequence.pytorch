"""
共通で使う関数などを置く
"""
import pickle
import torch
import numpy as np
import sys
from statistics import mean

# Cross Entropy
def calculate_mask(step_vector, vocab):
    return step_vector != vocab.pad()

def calculate_mask_NLL_loss(sentence_outputs, step_vector, mask):
    sentence_outputs = sentence_outputs.view(-1, sentence_outputs.size(-1))
    step_vector = step_vector.contiguous().view(-1, 1)
    losses = -torch.gather(sentence_outputs, dim=1, index=step_vector)
    mask = mask.view(-1, 1)
    losses = losses * mask.float()
    loss = losses.sum() / mask.float().sum()
    return loss