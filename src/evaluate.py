import argparse
import pickle
import os
import sys
import torch
sys.path.append("../utils")

from vocabulary import Vocabulary
from common import *
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

    generated_recipes = []
    refer_recipes = []
    for test_file in test_files[:100]:
        gen, refer = get_outputs(test_file, model, vocab)
        generated_recipes.append(gen)
        refer_recipes.append(refer)
    
    import ipdb; ipdb.set_trace()

    bleu1, bleu2, bleu3, bleu4 = calculate_BLEU(generated_recipes, refer_recipes)
    refer_recipes = [refer_recipe[0] for refer_recipe in refer_recipes]
    r_score = calculate_rouge(generated_recipes, refer_recipes)
    cider_D = calculate_CIDEr(generated_recipes, refer_recipes)
    meteor = calculate_METEOR(generated_recipes, refer_recipes)

    print("BLEU-1 : ", bleu1)
    print("BLEU-2 : ", bleu2)
    print("BLEU-3 : ", bleu3)
    print("BLEU-4 : ", bleu4)
    print("ROUGE-L : ", r_score)
    print("CIDEr-D : ", cider_D)
    print("METEOR : ", meteor)
