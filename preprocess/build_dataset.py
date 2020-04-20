"""
This script is used for building a dataset for training/testing recipe generation models.
1. Extracting recipes, which have all steps aligned with images (otherwise, we remove).
2. According to the [Chandu et al. ACL2020], if there are several images attached to a step, we randomly pick up one.
3. Dumping them to pickle.
"""
import argparse
import json
import os

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
        context = remove_introduction(context)
        for step_content in context:
            if len(step_content["step_images"]) == 0:
                perfectly_aligned = False
        if perfectly_aligned:
            recipe["context"] = context
            aligned_recipes.append(recipe)      
    return aligned_recipes

def pick_up_one_image():
    pass

def dump_to_pickle():
    pass

def build_dataset(args):
    instructables_filename = os.path.join(args.directory, "instructables.json")
    instructable_recipes = load_and_extract_recipes(instructables_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", "-d", required=True, help="directory of dumpled json files via story_boarding_of_a_recipe")
    args = parser.parse_args()
    build_dataset(args)