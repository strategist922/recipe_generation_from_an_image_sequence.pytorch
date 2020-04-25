from torch.utils.data import Dataset, DataLoader
import pickle
import os

class StoryBoardingDataset(Dataset):
    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.data_files = [os.path.join(input_dir, filename) for filename in os.listdir(input_dir)]
        self.idx_dict = {}
        for idx, recipe_id in enumerate(self.data_files):
            self.idx_dict[idx] = recipe_id
  
    def __len__(self):
        return len(self.data_files)
  
    def __getitem__(self, idx):
        data_path = self.idx_dict[idx]
        with open(data_path, "rb") as f:
            data = pickle.load(f)

        image_vectors = data["image_vector"]
        step_vectors = data["step_vector"]

        sample = {
            "image_vectors" : image_vectors,
            "step_vectors" : step_vectors,
        }

        return sample

def collate_fn(samples):
  image_vectors = [sample["image_vectors"] for sample in samples]
  step_vectors = [sample["step_vectors"] for sample in samples]
    
  return {
    "image_vectors" : image_vectors,
    "step_vectors" : step_vectors,
    }