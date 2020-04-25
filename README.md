# recipe_generation_from_an_image_sequence.pytorch
This repo is implementations of neural recipe generators using PyTorch.  
Now we implemented the following 5 models:
- Images2seq (https://www.aclweb.org/anthology/N16-1147v2.pdf)
- GLAC Net (https://arxiv.org/pdf/1805.10973.pdf)
- Retrieval Attention (RetAttn) (https://www.aclweb.org/anthology/W19-8650.pdf)
- SSiD (https://www.aclweb.org/anthology/P19-1606.pdf)
- SSiL (https://www.aclweb.org/anthology/P19-1606.pdf)    
**Note** We could not implement the SSiD and SSiL perfectly due to lack of details of a finite state machine (FSM).

# Requirements
1. Python 3.7
2. CUDA 10.2 and cuDNN v7.6
3. PyTorch 1.5.0
4. tall other required modules  
```
pip install -r requirements.txt
```

# Data Preparation
The dataset used in this repo is the story boarding dataset (https://www.aclweb.org/anthology/P19-1606.pdf).
As mentioned [here](https://github.com/khyathiraghavi/storyboarding_data/issues/3), the original scripts did not save the train/val/test splits. Thus, this scripts lead you to download the data from instructable.com and snapguide.com and split them into train/val/test datasets.

### 1. Downloading the story boarding dataset.
Follow [this repo](https://github.com/misogil0116/story_boarding_data).  
Then, Copy `instructables.json` and `snapguide.json` to `data/` directory.

### 2. Preprocessing the dataset.
The following scripts lead you to split the dataset with train/val/test.
```python
cd preprocess
python build_dataset.py -d ./data/ --dl
python convert_pickles_into_trainable_format.py -d ./data -o ./data/features/
```

### 3. Training the models.  
[WIP]

# Training and Validation

# Testing

# Citation
```
@misc{taichi19recipe,
    author = {Taichi Nishimura},
    title = {recipe_generation_from_an_image_sequence.pytorch},
    howpublished = {https://github.com/misogil0116/recipe_generation_from_an_image_sequence.pytorch},
    year = {2020}
}
```