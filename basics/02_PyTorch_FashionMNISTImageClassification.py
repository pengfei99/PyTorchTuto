import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from basics.source.FashionMNISTImageClassifier import FashionMNISTImageClassifier
from torch import nn

""" 2.1 Training Data Preparation 
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

""" 2.1.1 Load Fashion-MNIST 
Here is an example of how to load the Fashion-MNIST dataset from TorchVision. Fashion-MNIST is a dataset of Zalando’s 
article images consisting of of 60,000 training examples and 10,000 test examples. Each example comprises a 
28×28 grayscale image and an associated label from one of 10 classes.

We load the FashionMNIST Dataset with this method
"""


def download_data(local_data_store_path):
    training_data = datasets.FashionMNIST(
        # root is the path where the data is stored
        root=local_data_store_path,
        # train specifies training or test dataset,
        train=True,
        # download=True downloads the data from the Internet if it's not available at root.
        download=True,
        # transform and target_transform specify the feature and label transformations
        transform=ToTensor()
    )

    test_data = datasets.FashionMNIST(
        root=local_data_store_path,
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


""" 2.1.2 Preparing data for training with torch.utils.data.Dataset
Check the custom Dataset class in source/CustomDataset
"""

""" 2.1.3 Preparing your data for training with torch.utils.data.DataLoaders
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


""" 2.1.4 Transforme data
The features and labels of training data may not always fit in the model requirements.  We use transforms to perform 
some manipulation of the data and make it suitable for training.

All TorchVision datasets have two parameters:
- transform: modify the features
- target_transform: modify the labels
They both accept callables(function) containing the transformation logic. 

The torchvision.transforms module (https://pytorch.org/vision/stable/transforms.html) offers several 
commonly-used transforms out of the box.
"""


def download_data_and_transform(local_data_store_path):
    training_data = datasets.FashionMNIST(
        # root is the path where the data is stored
        root=local_data_store_path,
        # train specifies training or test dataset,
        train=True,
        # download=True downloads the data from the Internet if it's not available at root.
        download=True,
        # transform and target_transform specify the feature and label transformations
        # ToTensor
        transform=ToTensor(),
        # Lambda transforms apply any user-defined lambda function. Here, we define a function to turn the integer
        # into a one-hot encoded tensor. It first creates a zero tensor of size 10 (the number of labels in our dataset)
        # and calls scatter which assigns a value=1 on the index as given by the label y
        # For example label 5: "Sandal" will be transformed to tensor([0., 0., 0., 0., 0., 1., 0., 0., 0., 0.])
        target_transform=Lambda(
            lambda y: torch.zeros(10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
    )

    test_data = datasets.FashionMNIST(
        root=local_data_store_path,
        train=False,
        download=True,
        transform=ToTensor(),
        target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
    )
    return training_data, test_data


"""2.2 Build model(neural networks) in pytorch
Neural networks comprise of layers/modules that perform operations on data. The torch.nn namespace provides all the 
building blocks you need to build your own neural network. Every module in PyTorch subclasses the nn.Module. A neural 
network is a module itself that consists of other modules (layers). This nested structure allows for building and 
managing complex architectures easily.

In this tutorial, we will build a NN to classify images in the FashionMNIST dataset. 

2.2.1 Build model(nn) with different layers
Please check on basics.source.FashionMNISTImageClassifier to know how we build the model

2.2.2 Model parameters
Many layers inside a neural network are parameterized, i.e. have associated weights and biases that are optimized 
during training. In pytorch, all models subclass nn.Module which automatically tracks all fields defined inside the 
model object, and makes all parameters accessible using the model's parameters() or named_parameters() methods.

In this example, we iterate over each parameter, and print its size and a preview of its values.
"""


def select_hardware_for_training(device_name):
    if device_name == 'cpu':
        return 'cpu'
    elif device_name == 'gpu':
        return 'cuda' if (device_name == "") & torch.cuda.is_available() else 'cpu'
    else:
        print("Unknown device name, choose cpu as default device")
        return 'cpu'


def model_application_test(model, device):
    # generate a random tensor to test the model
    test_tensor = torch.rand(1, 28, 28, device=device)
    # intermediate result
    logits = model(test_tensor)
    # normalize the intermediate result to value between 0 and 1 by using softmax
    pred_probab = nn.Softmax(dim=1)(logits)
    # Get the max prediction probability value as final result
    y_pred = pred_probab.argmax(1)
    print(f"Predicted class: {y_pred}")


def show_model_parameters(model):
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")


""" 2.3 Setting hyper-parameters
Hyper-parameters are adjustable parameters that let you control the model optimization process. Different 
hyper-parameter values can impact model training and convergence rates. For more info about hyper-parameter tuning, 
go https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html

In our tutorial, we define the following hyper-parameters for training:
- Number of Epochs: the number of how many times to iterate over the dataset
- Batch Size: the number of data samples seen by the model in each epoch
- Learning Rate: how much to update models parameters at each batch/epoch. Smaller values yield slow learning speed, 
                 while large values may result in unpredictable behavior during training.
                 
2.3.1 Add an optimization loop
Once we set our hyper-parameters, we can then train and optimize our model with an optimization loop. Each iteration of 
the optimization loop is called an epoch.

Each epoch consists of two main parts:
- The Train Loop: iterate over the training dataset and try to converge to optimal parameters.
- The Validation/Test Loop: iterate over the test dataset to check if model performance is improving.

2.3.1.1 Loss function
When presented with some training data, our untrained network is likely not to give the correct answer. Loss function 
measures the degree of dissimilarity of obtained result to the target value, and it is the loss function that we 
want to minimize during training. To calculate the loss we make a prediction using the inputs of our given data 
sample and compare it against the true data label value. Common loss functions includes
- nn.MSELoss (Mean Square Error) for regression tasks
- nn.NLLLoss (Negative Log Likelihood) for classification
- nn.CrossEntropyLoss combines nn.LogSoftmax and nn.NLLLoss.

We pass our model's output logits to nn.CrossEntropyLoss, which will normalize the logits and compute the prediction error.

2.3.1.2 Optimization pass
Optimization is the process of adjusting model parameters to reduce model error in each training step. Optimization 
algorithms define how this process is performed (in this example we use Stochastic Gradient Descent). All optimization 
logic is encapsulated in the optimizer object. Here, we use the SGD optimizer; additionally, there are many different 
optimizers available in PyTorch:
- ADAM
- RMSProp
There are no best universal optimizer, each optimizer works better for specific kinds of models and data.

Inside the training loop, optimization happens in three steps:
1. Call optimizer.zero_grad() to reset the gradients of model parameters. Gradients by default add up; to prevent 
   double-counting, we explicitly zero them at each iteration.
2. Back-propagate the prediction loss with a call to loss.backwards(). PyTorch deposits the gradients of the 
   loss w.r.t. each parameter.
3. Once we have our gradients, we call optimizer.step() to adjust the parameters by the gradients collected in the 
   backward pass.
"""


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (features, label) in enumerate(dataloader):
        # Compute prediction and loss
        prediction = model(features)
        loss = loss_fn(prediction, label)

        # Backpropagation to optimize the model's parameter
        # reset the gradients of model parameters to zero at each iteration.
        optimizer.zero_grad()
        # Back-propagate the prediction loss
        loss.backward()
        # adjust the parameters by the gradients collected in the backward pass
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(features)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def main():
    data_path = "/tmp/pytorch/data"
    # Step 1: Prepare data
    # download data without transforming the label to one-hot code
    training_data1, test_data1 = download_data(data_path)

    # download data with label one-hot code transformation
    training_data2, test_data2 = download_data_and_transform(data_path)

    # view the data
    # show_image(training_data)

    # prepare data in patch via dataloader
    train_dataloader = prepare_data_with_dataloader(training_data1, 64, True)
    test_dataloader = prepare_data_with_dataloader(test_data1, 64, True)

    # iterate and show data in data loader
    # iterate_data_in_dataloader(train_dataloader)

    # Step2 : Build model (neuron network)
    device = select_hardware_for_training("gpu")

    # We create an instance of FashionMNISTImageClassifier, and move it to the device.
    model = FashionMNISTImageClassifier().to(device)
    # print it's structure
    print(f"Model's structure: \n {model}")

    # Apply model to a tensor
    # model_application_test(model, device)

    # print parameters of each layer in the model
    # show_model_parameters(model)

    # Step3: Setting hyper-parameters and training loop
    learning_rate = 1e-3
    batch_size = 64
    epochs = 20

    # create a loss function
    loss_fn = nn.CrossEntropyLoss()

    # create an optimizer.
    # The optimizer has tow arguments:
    # 1. model's parameters that need to be trained
    # 2. The learning rate.
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Step 4: start the training process
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
    print("Done!")


if __name__ == "__main__":
    main()
