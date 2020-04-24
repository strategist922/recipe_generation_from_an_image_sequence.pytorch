import argparse
import pickle
import os
import sys
import torch
sys.path.append("../utils")

from vocabulary import Vocabulary
from common import RecipeGenerator, RecipeEvaluator
from models.glacnet import BatchGLACNet

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", "-d", type=str, required=True, help="directory path")
    parser.add_argument("--models", "-m", type=str, required=True, help="model path")
    parser.add_argument("--hidden_size", "-hs", default=512, help="hidden size")
    parser.add_argument("--n_layer", "-nl", default=1, help="the number of layers")
    args = parser.parse_args()
  
    directory = args.directory
    model_path = args.models
    hidden_size = args.hidden_size
    n_layer = args.n_layer
  
    vocab_dir = os.path.join(directory, "vocab.pkl")
    test_dir = os.path.join(directory, "features/test")
    test_files = [os.path.join(test_dir, filename) for filename in os.listdir(test_dir)]
    with open(vocab_dir, "rb") as f:
        vocab = pickle.load(f)
  
    img_feature_size = 512
    enc_hidden_size = hidden_size
    dec_hidden_size = hidden_size
    embed_size = hidden_size
    n_layer = n_layer
    vocab_size = len(vocab)

    model = BatchGLACNet(img_feature_size=512,
                        enc_hidden_size=hidden_size,
                        vocab_size=vocab_size,
                        embed_size=hidden_size,
                        dec_hidden_size=hidden_size)

    checkpoint = torch.load(args.models)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    generator = RecipeGenerator(model, vocab)
    generated_recipes = []
    refer_recipes = []

    for test_file in test_files:
        gen, refer = generator.generate(test_file)
        generated_recipes.append(gen)
        refer_recipes.append(refer)

    evaluator = RecipeEvaluator(generated_recipes, refer_recipes)
    evaluator.evaluate()