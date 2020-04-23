"""
Training scripts
"""
import argparse
import torch
import pickle
import os
from torch import optim
import torch.nn as nn
import numpy as np

import sys
sys.path.append("../utils")
from vocabulary import Vocabulary

from torch.utils.data import DataLoader
from dataset_utils import StoryBoardingDataset, collate_fn
from models.glacnet import BatchGLACNet
from common import calculate_mask_NLL_loss, calculate_mask

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

def evaluate(model, batch_size, vocab):
    model.eval()
    pad_num = vocab_size - 1

    dataset = StoryBoardingDataset(input_dir=os.path.join(directory, "features/val"))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loss = 0
    count_c = 0

    for i, sample_batched in enumerate(dataloader):
        step_vectors, image_vectors = sample_batched["step_vectors"], sample_batched["image_vectors"]
        batch_image_vectors, batch_step_vectors = batchfy(image_vectors, step_vectors, pad_num)
        sent_output = model(batch_step_vectors, batch_image_vectors, pad_num)
        mask = calculate_mask(batch_step_vectors, vocab)
        loss = calculate_mask_NLL_loss(sent_output, batch_step_vectors, mask)
        val_loss += loss.item()
        count_c += 1
    val_loss = val_loss/count_c
    return val_loss

def batchfy(image_vectors, step_vectors, pad_num):
    max_seq_length = max([image_vector.shape[0] for image_vector in image_vectors])
    batched_image_vectors = np.zeros((batch_size, max_seq_length, 512))
    for img_idx, image_vector in enumerate(image_vectors):
        seq_length = image_vector.shape[0]
        batched_image_vectors[img_idx, :seq_length, :] = image_vector#.detach().numpy()
    
    max_T = max([step_vector.shape[1] for step_vector in step_vectors])
    batched_step_vectors = np.full((batch_size, max_seq_length, max_T), pad_num)
    for stp_idx, step_vector in enumerate(step_vectors):
        seq_length, sent_length = step_vector.shape
        batched_step_vectors[stp_idx, :seq_length, :sent_length] = step_vector

    truncated_image_length = 20
    truncated_sent_length = 85

    batched_image_vectors = batched_image_vectors[:, :truncated_image_length]
    batched_step_vectors = batched_step_vectors[:, :truncated_image_length, :truncated_sent_length]

    batched_image_vectors = torch.Tensor(batched_image_vectors).to(device)
    batched_step_vectors = torch.LongTensor(batched_step_vectors).to(device)
    return batched_image_vectors, batched_step_vectors

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", "-d", default="./data", help="training file directory")
    parser.add_argument("--model", "-m", required=True, help="the name of model you want to train: glacnet, images2seq, retattn, ssid or ssil")
    parser.add_argument("--clip", "-c", default=50, help="clip gradients if gradient norm is over the value")
    parser.add_argument("--iteration", "-itr", default=30, help="the number of iteration")
    parser.add_argument("--hidden_size", "-hs", default=512, help="hidden size")
    parser.add_argument("--learning_rate", "-lr", default=0.001, help="learning rate")
    parser.add_argument("--n_layer", "-nl", default=1, help="the number of layers")
    parser.add_argument("--batch_size", "-bs", default=16, type=int, help="batch size")
    args = parser.parse_args()
    
    directory = args.directory
    clip = args.clip
    n_iteration = args.iteration
    hidden_size = args.hidden_size
    n_layer = args.n_layer
    learning_rate = args.learning_rate
    model_name = args.model
    batch_size = args.batch_size

    with open(os.path.join(directory, "vocab.pkl"), "rb") as f:
        vocab = pickle.load(f)
    
    vocab_size = len(vocab)
    pad_num = vocab_size - 1
    best_loss = 10000
    best_epoch = -1
    early_stopping = 0

    dataset = StoryBoardingDataset(input_dir=os.path.join(directory, "features/train"))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)

    model = BatchGLACNet(img_feature_size=512,
                        enc_hidden_size=hidden_size,
                        vocab_size=vocab_size,
                        embed_size=hidden_size,
                        dec_hidden_size=hidden_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)

    for epoch in range(n_iteration):
        epoch_loss = 0
        epoch_num = 0
        model.train()

        for i, sample_batched in enumerate(dataloader):
            loss = 0
            step_vectors, image_vectors = sample_batched["step_vectors"], sample_batched["image_vectors"]
            batch_image_vectors, batch_step_vectors = batchfy(image_vectors, step_vectors, pad_num)
            sent_output = model(batch_step_vectors, batch_image_vectors, pad_num)
            mask = calculate_mask(batch_step_vectors, vocab)
            loss = calculate_mask_NLL_loss(sent_output, batch_step_vectors, mask)
            
            # back propargation
            optimizer.zero_grad()
            loss.backward()

            # clip gradient
            _ = nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            # epoch_loss
            epoch_loss += loss.item()
            epoch_num += 1
            print("epoch :[{}/{}] Iter :[{}/{}] Total loss : {:.4f}".format(epoch, n_iteration, i, len(dataloader), loss.item()))
        print("=== Epoch : {} total_loss : {:.4f} ===".format(epoch, epoch_loss/epoch_num))
        
        # Evaluation w/ validation data
        val_loss = evaluate(model, batch_size, vocab)

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            filename = os.path.join(directory, "models", model_name, "model_{}_{:.4f}.t7".format(best_epoch, best_loss))
            torch.save(model.state_dict(), filename)
        else:
            early_stopping += 1
        
        if early_stopping >= 3:
            print("stopped best_epoch {} best loss {:.4f}".format(best_epoch, best_loss))
            break