import os
import pandas as pd
import torchvision.io as tvio
from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    # The __init__ function is run once when instantiating the CustomImageDataset object. We initialize the directory
    # containing the images, the annotations file, and both transforms
    # The annotations_file looks like this:
    #     image_name | label
    #     img1.jpg, 0
    #     img2.jpg, 0
    #     ......
    #     ankleboot999.jpg, 9
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    # The __len__ function returns the number of samples in our dataset.
    def __len__(self):
        return len(self.img_labels)

    # The __getitem__ function loads and returns a sample from the dataset at the given index idx. Based on the index,
    # it identifies the image's location on disk, converts that to a tensor using read_image, retrieves the
    # corresponding label from the csv data in self.img_labels, calls the transform functions on them (if applicable),
    # and returns the tensor image and corresponding label in a Python dict.
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = tvio.read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        sample = {"image": image, "label": label}
        return sample
