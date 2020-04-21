"""
This script is used for building a dataset and vocabulary for training/testing recipe generation models.
1. Extracting recipes, which have all steps aligned with images (otherwise, we remove).
2. According to the [Chandu et al. ACL2020], if there are several images attached to a step, we randomly pick up one.
3. Downloading images
4. Spliting 8:1:1 (train/val/split)
5. Building vocabulary using training data
6. Dumpling data to pickle files
"""
import argparse
import json
import os
import random
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

def pick_up_one_image():
    pass

def dump_to_pickle():
    pass

def build_dataset(args):
    # Instructables.com
    instructables_filename = os.path.join(args.directory, "instructables.json")
    instructable_recipes = load_and_extract_recipes(instructables_filename)

    # snapguide.com
    snapguide_filename = os.path.join(args.directory, "snapguide.json")
    snapguide_recipes = load_and_extract_recipes(snapguide_filename)
    #import ipdb; ipdb.set_trace()

    # download images
    recipes = instructable_recipes + snapguide_recipes
    if args.dl:
        image_dir = os.path.join(args.directory, "images")
        download_images(image_dir, recipes)

    # [WIP]
    # Split them into 8:1:1
    
    # Building vocabulary with training data

    # Saving training, val, test, and vocabulary data.
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", "-d", required=True, help="directory of dumpled json files via story_boarding_of_a_recipe")
    parser.add_argument("--dl", action='store_true', help="True if you want to download")
    args = parser.parse_args()
    build_dataset(args)