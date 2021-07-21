import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

""" 2 Training Data Preparation 
Code for processing data samples can get messy and hard to maintain. Normally we separate our data preparation code 
from our model training code. This can increase readability and modularity of our code. 

PyTorch provides two data primitives: 
- torch.utils.data.DataLoader
- torch.utils.data.Dataset
They allow you to use pre-loaded datasets as well as your own data. Dataset stores the samples and their corresponding 
labels, and DataLoader wraps an iterable around the Dataset to enable easy access to the samples.


PyTorch domain libraries provide a number of pre-loaded datasets (such as FashionMNIST) that subclass 
torch.utils.data.Dataset and implement functions specific to the particular data. They can be used to prototype 
and benchmark your model. You can find them here: 
- Image Datasets: https://pytorch.org/vision/stable/datasets.html
- Text Datasets: https://pytorch.org/text/stable/datasets.html
- Audio Datasets: https://pytorch.org/audio/stable/datasets.html


"""

""" 2.1 Load Fashion-MNIST 
Here is an example of how to load the Fashion-MNIST dataset from TorchVision. Fashion-MNIST is a dataset of Zalando’s 
article images consisting of of 60,000 training examples and 10,000 test examples. Each example comprises a 
28×28 grayscale image and an associated label from one of 10 classes.

We load the FashionMNIST Dataset with this method
"""


def download_data():
    data_path = "/tmp/pytorch/data"
    training_data = datasets.FashionMNIST(
        # root is the path where the data is stored
        root=data_path,
        # train specifies training or test dataset,
        train=True,
        # download=True downloads the data from the Internet if it's not available at root.
        download=True,
        # transform and target_transform specify the feature and label transformations
        transform=ToTensor()
    )

    test_data = datasets.FashionMNIST(
        root=data_path,
        train=False,
        download=True,
        transform=ToTensor()
    )
    return training_data, test_data


def show_image(data):
    labels_map = {
        0: "T-Shirt",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot",
    }

    figure = plt.figure(figsize=(8, 8))
    cols = 3
    rows = 3
    for i in range(1, cols * rows + 1):
        # torch.randint returns a tensor filled with random integers generated uniformly between
        # low (inclusive) and high (exclusive).
        # The shape of the tensor is defined by the variable argument size, in our case (1,) it's one dimension which
        # has 1 element. In another word it is a simple int.
        # For example, torch.randint(10, (2, 2)) returns tensor([[0, 2],[5, 5]]), two dimension and
        # each dimension has two element.
        sample_idx = torch.randint(len(data), size=(1,)).item()
        img, label = data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()


""" 2.2 Preparing data for training with torch.utils.data.Dataset
Check the custom Dataset in source/CustomDataset
"""

""" 2.3 Preparing your data for training with torch.utils.data.DataLoaders
The Dataset retrieves our dataset's features and labels one sample at a time. While training a model, we typically 
want to pass samples in "mini batches", reshuffle the data at every epoch to reduce model over-fitting, and use 
Python's multiprocessing to speed up data retrieval.

DataLoader is an iterable that abstracts this complexity for us in an easy API.

"""


def prepare_data_with_dataloader(data, batch_size: int, shuffle: bool):
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle)


def iterate_data_in_dataloader(data_in_dataloader):
    # With the prepare_data_with_dataloader method, we have loaded that dataset into the Dataloader and can iterate
    # through the dataset as needed. Each iteration of the dataloader returns a batch of train_features and train_labels
    #
    # (containing batch_size=64 features and labels respectively).
    # Because we specified shuffle=True, after we iterate over all batches the data is shuffled
    # Display image and label.
    train_features, train_labels = next(iter(data_in_dataloader))
    # Note the shape of feature is four dimensions [64, 1, 28, 28]
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze()
    label = train_labels[0]
    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"Label: {label}")


def main():
    # Step 1: Prepare data
    # download data
    training_data, test_data = download_data()

    # view the data
    # show_image(training_data)

    # prepare data in patch via dataloader
    train_dataloader = prepare_data_with_dataloader(training_data, 64, True)
    test_dataloader = prepare_data_with_dataloader(test_data, 64, True)

    # iterate and show data in data loader
    iterate_data_in_dataloader(train_dataloader)


if __name__ == "__main__":
    main()
