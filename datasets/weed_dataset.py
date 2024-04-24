import os
import torch
from PIL import Image
import pandas as pd

class WeedDataset(torch.utils.data.Dataset):

  def __init__(self, crop, transform = None):

    self.transform = transform

    if crop == "deepweeds":
      image_dir = "deepweeds/images"
      df = pd.read_csv("deepweeds/labels.csv")

      self.imagepaths = [os.path.join(image_dir, imagename)
                            for imagename in df["Filename"]]
      self.labels = list(df["Label"])
      self.categories = list(df["Species"])

      self.category2index = {}

      for i in range(len(self.categories)):
        if self.categories[i] not in self.category2index.keys():
          self.category2index[self.categories[i]] = self.labels[i]

    else:

      self.crop_dir = "weed-datasets/" + crop +"/"

      self.categories = os.listdir(self.crop_dir)

      self.category2index = {category: idx for (idx, category) in enumerate(self.categories)}

      # Compile a list of images and corresponding labels.
      self.imagepaths = []
      self.labels = []
      for category in self.categories:
        category_directory = self.crop_dir + category
        category_imagenames = os.listdir(category_directory)

        self.imagepaths += [os.path.join(category_directory, imagename)
                            for imagename in category_imagenames]
        self.labels += [self.category2index[category]] * len(category_imagenames)

    # Sort imagepaths alphabetically and labels accordingly.
    sorted_pairs = sorted(zip(self.imagepaths, self.labels), key = lambda x: x[0])
    self.imagepaths, self.labels = zip(*sorted_pairs)

  # Return a sample (x, y) as a tuple e.g. (image, label)
  def __getitem__(self, index):
    image = Image.open(self.imagepaths[index]).convert('RGB')
    if self.transform:
      image = self.transform(image)
    return image, self.labels[index]

  # Return the total number of samples.
  def __len__(self):
    return len(self.imagepaths)