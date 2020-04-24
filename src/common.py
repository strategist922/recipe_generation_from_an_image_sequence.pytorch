"""
共通で使う関数などを置く
"""
import pickle
import torch
import numpy as np
import sys
import nltk
from rouge import Rouge
from statistics import mean

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

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

# Generating Recipes
class RecipeGenerator:
    def __init__(self, model, vocab):
        self.model = model
        self.vocab = vocab
        self.id2word_dict = self.id_to_word()
    
    def generate(self, test_file):
        """
        Generate a recipe w/ the model
        """
        sent_out, refer_out = self.get_predict_and_reference(test_file)
        sent_out = self.flatten_recipe(self.sent_ids_to_words(sent_out))
        refer_out = [ self.flatten_recipe(self.sent_ids_to_words(refer_out)) ]
        return sent_out, refer_out

    def flatten_recipe(self, recipe):
        recipe_words = []
        for step in recipe:
            for word in step:
                if word == "<SOS>" or word == "<EOS>":
                    continue
                recipe_words.append(word)
        return recipe_words

    def sent_ids_to_words(self, out_vectors):
        outputs = []
        for out_vector in out_vectors:
            output = [self.id2word_dict[wid] for wid in out_vector]
            outputs.append(output)
        return outputs

    def get_predict_and_reference(self, test_file):
        with open(test_file, "rb") as f:
            data = pickle.load(f)
        
        images = data["image_vector"]
        referece_vector = torch.LongTensor(data["step_vector"])
        
        images = torch.Tensor(images).to(device)
        sent_output = self.model.predict(images, self.vocab)
        ref_out = self.mask_pad(referece_vector)
        return sent_output, ref_out
    
    def mask_pad(self, referece_vector):
        pad_id = self.vocab.pad()
        ref_outs = []
        for r_sent in referece_vector:
            ref_outs.append(r_sent[r_sent != pad_id].numpy().tolist())
        return ref_outs

    def id_to_word(self):
        id2word_dict = {v:k for k, v in self.vocab._vocab.items()}
        id2word_dict[self.vocab.pad()] = "<PAD>"
        id2word_dict[self.vocab.sos()] = "<SOS>"
        id2word_dict[self.vocab.eos()] = "<EOS>"
        id2word_dict[self.vocab._unk_id] = "<UNK>"
        return id2word_dict

# Evaluation
class RecipeEvaluator:
    def __init__(self, generated_recipes, reference_recipes):
        self.generated_recipes = generated_recipes
        self.reference_recipes = reference_recipes

    def evaluate(self):
        self.clean_recipe()
        bleu1, _, _, bleu4 = self.calculate_BLEU()
        self.reference_recipes = [reference_recipe[0] for reference_recipe in self.reference_recipes]
        # meteor = self.calculate_METEOR()
        rouge_l = self.calculate_rouge()

        print("BLEU-1 : ", bleu1)
        print("BLEU-4 : ", bleu4)
        print("ROUGE-L : ", rouge_l)
        # print("CIDEr-D : ", cider_D): TODO
        # print("METEOR : ", meteor)

    def clean_recipe(self):
        self.generated_recipe = [word for word in self.generated_recipes if word != "\n"]
        

    def calculate_BLEU(self):
        bleu1 = nltk.translate.bleu_score.corpus_bleu(self.reference_recipes, self.generated_recipes, weights=[1.0, 0, 0, 0])
        bleu2 = nltk.translate.bleu_score.corpus_bleu(self.reference_recipes, self.generated_recipes, weights=[0.5, 0.5, 0, 0])
        bleu3 = nltk.translate.bleu_score.corpus_bleu(self.reference_recipes, self.generated_recipes, weights=[1/3, 1/3, 1/3, 0])
        bleu4 = nltk.translate.bleu_score.corpus_bleu(self.reference_recipes, self.generated_recipes)
        return bleu1, bleu2, bleu3, bleu4

    def calculate_METEOR(self):
        meteors = []
        for generated_recipe, refer_recipe in zip(self.generated_recipes, self.reference_recipes):
            import ipdb; ipdb.set_trace()
            generated_recipe = " ".join(generated_recipe)
            refer_recipe = " ".join(refer_recipe)
            meteor = nltk.translate.meteor_score.single_meteor_score(refer_recipe, generated_recipe)
            meteors.append(meteor)
        meteor_score = mean(meteors)
        return meteor_score

    def calculate_rouge(self):
        rouge = Rouge()
        generated_recipes = [" ".join(generated_recipe) for generated_recipe in self.generated_recipes]
        refer_recipes = [" ".join(refer_recipe) for refer_recipe in self.reference_recipes]
        r_length = 0
        sum_score = 0
        for generated_recipe, refer_recipe in zip(generated_recipes, refer_recipes):
            try:
                sum_score += rouge.get_scores(generated_recipe, refer_recipe)[0]["rouge-l"]["f"]
                r_length += 1
            except:
                continue
        if r_length == 0:
            print("rouge-L 0")
            score = 0
        else:
            score = sum_score / r_length
        return score
