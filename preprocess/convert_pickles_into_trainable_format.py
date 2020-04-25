"""
This script helps to convert data into trainable format.
"""
import argparse
import pickle
import os
import numpy as np
import random
import sys
sys.path.append("../utils")
from tqdm import tqdm

# for text2vec
from vocabulary import Vocabulary

# for image2vec
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

IMAGE_DIMENSION_SIZE = 512

def load_pickle(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data

def text2vec(vocab, recipes):
    out_recipes = []
    for recipe in recipes:
        seq_length = len(recipe["context"])
        context = recipe["context"]
        if len(context) == 0:
            continue
        max_token_length = max([len(step["token"]) for step in context])
        recipe_vector = np.full((seq_length, max_token_length+2), vocab.pad()) #NOTE: +2 is for <sos> and <eos>
        for idx, step in enumerate(context):
            token = step["token"]
            step_vector = np.array([vocab.sos()] + [vocab.word_to_id(word) for word in token] + [vocab.eos()])
            step_len = step_vector.shape[0]
            recipe_vector[idx, :step_len] = step_vector
        recipe["step_vector"] = recipe_vector
        out_recipes.append(recipe)
    return out_recipes

def image_pickup_an_image_randomly(files):
    choosen_file = random.choice(files)
    return choosen_file[1]

def extract_feature_vector(cnn_model, filename):
    try:
        transformer = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image = Image.open(filename)
        image = transformer(image)
        image_vector = cnn_model(image.unsqueeze(0))
        return image_vector.flatten()
    except:
        return None

def image2vec(vocab, input_dir, recipes):
    cnn_model = models.resnet34(pretrained=True)
    cnn_model = nn.Sequential(*list(cnn_model.children())[:-1])
    cnn_model.eval()
    image_root_dir = os.path.join(input_dir, "images")
    out_recipes = [] 
    for recipe in tqdm(recipes):
        context = recipe["context"]
        image_vectors = np.zeros((len(context), IMAGE_DIMENSION_SIZE))
        is_image_ok = True
        for idx, step in enumerate(context):
            filename = image_pickup_an_image_randomly(step["step_images"])
            filename = os.path.join(image_root_dir, filename)
            
            # convert image into feature vector w/ ResNet34
            image_feature = extract_feature_vector(cnn_model, filename)
            if image_feature is None:
                is_image_ok = False
                break
            image_vectors[idx] = image_feature.detach().numpy()
        
        if is_image_ok:
            recipe["image_vector"] = image_vectors
            out_recipes.append(recipe)
    return out_recipes

def convert_data_to_vectors(vocab, recipes, input_dir, output_dir):
    """
    # text2vec
    1. convert tokens into word ids with vocabulary
    2. insert padding id into token
    # image2vec
    1. convert an image into a feature vector
    """
    recipes = text2vec(vocab, recipes)
    recipes = image2vec(vocab, input_dir, recipes)

    for idx, recipe in enumerate(recipes):
        filename = os.path.join(output_dir, str(idx) + ".pkl")
        with open(filename, "wb") as f:
            pickle.dump(recipe, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", "-d", required=True, help="directory of dumpled json files via story_boarding_of_a_recipe")
    parser.add_argument("--output_dir", "-o", required=True, help="output directory for serialization")
    args = parser.parse_args()

    vocab_dir = os.path.join(args.directory, "vocab.pkl")
    train_dir = os.path.join(args.directory, "train_recipes.pkl")
    val_dir = os.path.join(args.directory, "val_recipes.pkl")
    test_dir = os.path.join(args.directory, "test_recipes.pkl")

    vocab = load_pickle(vocab_dir)
    train_recipes = load_pickle(train_dir)
    val_recipes = load_pickle(val_dir)
    test_recipes = load_pickle(test_dir)
    
    out_train_dir = os.path.join(args.output_dir, "train")
    out_val_dir = os.path.join(args.output_dir, "val")
    out_test_dir = os.path.join(args.output_dir, "test")

    convert_data_to_vectors(vocab, train_recipes, args.directory, out_train_dir)
    convert_data_to_vectors(vocab, val_recipes, args.directory, out_val_dir)
    convert_data_to_vectors(vocab, test_recipes, args.directory, out_test_dir)