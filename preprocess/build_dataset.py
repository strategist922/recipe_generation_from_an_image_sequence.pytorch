"""
This script is used for building a dataset and vocabulary for training/testing recipe generation models.
1. Extracting recipes, which have all steps aligned with images (otherwise, we remove).
2. According to the [Chandu et al. ACL2020], if there are several images attached to a step, we randomly pick up one.
3. Downloading images
4. Spliting 8:1:1 (train/val/split)
5. Building vocabulary using training data
6. Dumpling data to pickle files
"""
import pickle
import argparse
import json
import os
import random
import spacy
from tqdm import tqdm
import sys
sys.path.append("../utils")
from vocabulary import Vocabulary
from collections import Counter
random.seed(42)

def remove_introduction(context):
    if "introduction" in context[0]["step_title"].lower():
            return context[1:]      
    else:
        return context

def load_and_extract_recipes(filename):
    with open(filename, "r") as f:
        recipes = json.load(f)
    aligned_recipes = []
    for recipe in recipes:
        context = recipe["context"]
        perfectly_aligned = True

        if len(context) == 0:
            continue

        context = remove_introduction(context)
        for step_content in context:
            if len(step_content["step_images"]) == 0:
                perfectly_aligned = False
        if perfectly_aligned:
            recipe["context"] = context
            aligned_recipes.append(recipe)      
    return aligned_recipes

def download_images(image_dir, recipes):
    for recipe in recipes:
        context = recipe["context"]
        for step in context:
            images = step["step_images"]
            for image in images:
                url, filename = image[0], image[1]
                filename = os.path.join(image_dir, filename)
                os.system("wget -O "+ filename + " " + url)

def tokenize(recipes):
    """
    Two preprocesses:
    1. lower() method
    2. tokenize w/ spacy
    """
    spacy_model = spacy.load('en')
    for recipe in tqdm(recipes):
        context = recipe["context"]
        for step in context:
            step_text = step["step_text"].lower()
            output = [tok.text for tok in spacy_model.tokenizer(step_text)]
            step["token"] = output
    return recipes

def dump_to_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)

def build_vocabulary(train_recipes):
    """
    Building vocabulary.
    In our case, we replace words with <unk> when they have less than 5 frequency.
    """
    counter = Counter()

    for recipe in train_recipes:
        context = recipe["context"]
        for step in context:
            token = step["token"]
            counter.update(token)
    
    min_count = 5
    word_counts = [x for x in counter.items() if x[1] >= min_count]
    word_counts.sort(key=lambda x : x[1], reverse=True)
    print("# Words in Vocabulary : ", len(word_counts))
    reverse_vocab = [x[0] for x in word_counts]

    unk_id = len(reverse_vocab)
    sos_id = len(reverse_vocab) + 1
    eos_id = len(reverse_vocab) + 2
    pad_id = len(reverse_vocab) + 3

    vocab_dict = dict([(x, y) for (y, x) in enumerate(reverse_vocab)])
    vocab = Vocabulary(vocab_dict, unk_id, sos_id, eos_id, pad_id)
    return vocab

def build_dataset(args):
    # Instructables.com
    instructables_filename = os.path.join(args.directory, "instructables.json")
    instructable_recipes = load_and_extract_recipes(instructables_filename)

    # snapguide.com
    snapguide_filename = os.path.join(args.directory, "snapguide.json")
    snapguide_recipes = load_and_extract_recipes(snapguide_filename)
    
    import ipdb; ipdb.set_trace()

    # download images
    recipes = instructable_recipes + snapguide_recipes
    if args.dl:
        image_dir = os.path.join(args.directory, "images")
        download_images(image_dir, recipes)

    # Tokenizer w/ spacy
    recipes = tokenize(recipes)

    # Split them into 8:1:1
    random.shuffle(recipes)
    train_recipes, val_test_recipes = recipes[:int(len(recipes)*0.8)], recipes[int(len(recipes)*0.8):]
    val_recipes, test_recipes = val_test_recipes[:int(len(val_test_recipes)*0.5)], val_test_recipes[int(len(val_test_recipes)*0.5):]

    # Building vocabulary with training data
    vocab = build_vocabulary(train_recipes)

    # Saving training, val, test, and vocabulary data
    vocab_filename = os.path.join(args.directory, "vocab.pkl")
    train_filename = os.path.join(args.directory, "train_recipes.pkl")
    val_filename = os.path.join(args.directory, "val_recipes.pkl")
    test_filename = os.path.join(args.directory, "test_recipes.pkl")
    
    dump_to_pickle(vocab, vocab_filename)
    dump_to_pickle(train_recipes, train_filename)
    dump_to_pickle(val_recipes, val_filename)
    dump_to_pickle(test_recipes, test_filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", "-d", required=True, help="directory of dumpled json files via story_boarding_of_a_recipe")
    parser.add_argument("--dl", action='store_true', help="True if you want to download")
    args = parser.parse_args()
    build_dataset(args)
