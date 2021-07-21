import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt

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


def load_data():
    training_data = datasets.FashionMNIST(
        # root is the path where the data is stored
        root="data",
        # train specifies training or test dataset,
        train=True,
        # download=True downloads the data from the Internet if it's not available at root.
        download=True,
        # transform and target_transform specify the feature and label transformations
        transform=ToTensor()
    )

    test_data = datasets.FashionMNIST(
        root="data",
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


def main():
    # Step 1: Prepare data
    training_data, test_data = load_data()
    show_image(training_data)


if __name__ == "__main__":
    main()
