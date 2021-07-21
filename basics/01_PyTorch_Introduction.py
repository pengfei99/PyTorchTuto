import torch
import numpy as np

""" 1. PyTorch Introduction
PyTorch is developed by Facebook AI Research (FAIR). PyTorch is one of the most widely used open-source machine
learning libraries for deep learning applications. It was first introduced in 2016. Since then, PyTorch has been
gaining popularity among researchers and developers, at the expense of TensorFlow.

Machine learning workflows involve
- preparing training data
- creating models
- optimizing model parameters
- saving the trained models.

In this tutorial you will:
- Learn the key concepts used to build machine learning models
- Learn how to build a Computer Vision model
- Build models with the PyTorch API
"""

"""  1.1 Tensors

Tensors are a specialized data structure that are very similar to arrays and matrices. In PyTorch, we use tensors to 
encode the inputs and outputs of a model, as well as the model’s parameters.

Tensors are similar to NumPy’s ndarrays, except that tensors can run on GPUs or other hardware accelerators. In fact, 
tensors and NumPy arrays can often share the same underlying memory, eliminating the need to copy data 
(see bridge-to-np-label). Tensors are also optimized for automatic differentiation 
(we'll see more about that later in the Autograd unit). If you’re familiar with ndarrays, you’ll be right at home 
with the Tensor API. If not, follow along!
"""

""" 1.1.1 Tensors creation
Tensors can be initialized in various ways:
- directly from data: The data type is automatically inferred.
- from a numpy array: np array to tensor and vice-versa is completely automatic
- from other tensor: The new tensor retains the properties (shape, data type) of the argument tensor, 
                     unless explicitly overridden. 
- from a shape and values:  
"""


def tensor_creation():
    data = [[1, 2], [3, 4]]
    # 1. create tensor from data
    tensor_from_data = torch.tensor(data)
    print(f"tensor's shape: {tensor_from_data.size()}")

    # 2. create tensor from np array
    np_array = np.array(data)
    tensor_from_np = torch.from_numpy(np_array)

    # 3. create tensor from other tensor
    # retains the properties of tensor_from_data
    int_tensor = torch.ones_like(tensor_from_data)
    print(f"Random Tensor has the same shape and type of tensor_from_data: \n {int_tensor} \n")

    # retains the shape but overrides the datatype with float
    float_tensor = torch.rand_like(tensor_from_np, dtype=torch.float)
    print(f"Random Tensor has the same shape of tensor_from_np, but with float type : \n {float_tensor} \n")

    # 4. create tensor with shape and values
    shape = (2, 3,)
    # use the shape, fill with random value
    rand_tensor = torch.rand(shape)
    print(f"Random Tensor: \n {rand_tensor} \n")
    # use the shape, fill with 1
    ones_tensor = torch.ones(shape)
    print(f"Ones Tensor: \n {ones_tensor} \n")
    # use the shape, fill with 0
    zeros_tensor = torch.zeros(shape)
    print(f"Zeros Tensor: \n {zeros_tensor}")


""" 1.1.2 Tensor attributs
Tensor has three attributes:
- shape
- datatype
- the device on which they are stored
"""


def tensor_attributes():
    tensor = torch.rand(3, 4)
    print(f"Shape of tensor: {tensor.shape}")
    print(f"Datatype of tensor: {tensor.dtype}")
    print(f"Device tensor is stored on: {tensor.device}")


"""1.1.3 Tensor Operations
PyTorch provides over 100 tensor operations, including:
- arithmetic
- linear algebra
- matrix manipulation (transposing, indexing, slicing)
- sampling 
- Etc. 
You can find more details here https://pytorch.org/docs/stable/torch.html
"""


def tensor_operations():
    # All operations can be run on the GPU (at typically higher speeds than on a CPU).
    # By default, tensors are created on the CPU. We need to explicitly move tensors to the GPU using .to method
    # (after checking for GPU availability). Keep in mind that copying large tensors across devices can be expensive
    # in terms of time and memory!

    # We move our tensor to the GPU if available
    tensor = torch.rand(3, 4)
    if torch.cuda.is_available():
        tensor = tensor.to('cuda')
    else:
        print("No GPU found")

    # Tensor slicing
    print("Full tensor content: \n", tensor)
    print('First row: ', tensor[0])
    print('First column: ', tensor[:, 0])
    print('Last column:', tensor[..., -1])
    # Change tensor value.
    tensor[:, 1] = 0
    print("Full tensor content, after value modification on column 2: \n", tensor)

    # Joining tensors
    # You can use torch.cat to concatenate a sequence of tensors along a given dimension. See also torch.stack,
    # another tensor joining op that is subtly different from torch.cat.
    concat_tensor = torch.cat([tensor, tensor, tensor], dim=1)
    # shape is the field of object tensor, size() is the function that returns this field
    print("Concat tensors shape: ", concat_tensor.shape)
    print("Concat tensors result: \n", concat_tensor)

    # arithmetic operations
    # This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
    y1 = tensor @ tensor.T
    y2 = tensor.matmul(tensor.T)
    y3 = torch.rand_like(tensor)
    torch.matmul(tensor, tensor.T, out=y3)

    # single element tensor
    # If you have a one-element tensor, for example by aggregating all values of a tensor into one value, you can
    # convert it to a Python numerical value using item():
    agg_tensor = tensor.sum()
    agg_item = agg_tensor.item()
    print(f"agg_tensor type: {type(agg_tensor)},agg_tensor value: {agg_tensor}")
    print(f"agg_item type: {type(agg_item)},agg_item value: {agg_item}")

    # In-place operations
    # Operations that store the result into the operand are called in-place. They are denoted by a _ suffix.
    # For example: x.copy_(y), x.t_(), will change x
    print("Source tensor: \n", tensor, "\n")
    # add 5 to all values in tensor
    tensor.add_(5)
    print("Result tensor after add_(5): \n", tensor)

    # This computes the element-wise product. z1, z2, z3 will have the same value
    z1 = tensor * tensor
    z2 = tensor.mul(tensor)
    z3 = torch.rand_like(tensor)
    torch.mul(tensor, tensor, out=z3)


""" Bridge with NumPy
Tensors on the CPU and NumPy arrays can share their underlying memory locations, and changing one will change the other.
"""


def tensor_np_conversion():
    # create a tensor and convert it to np array
    tensor = torch.ones(5)
    print(f"source tensor: {tensor}")
    np_array = tensor.numpy()
    print(f"convert to numpy array: {np_array}")

    # change the tensor value reflects in the NumPy array.
    tensor.add_(7)
    print(f"tensor value after add_(7): {tensor}")
    print(f"np array value after add_(7): {np_array}")

    # create a np array and convert it to tensor
    np_array1 = np.ones(3)
    tensor1 = torch.from_numpy(np_array1)
    print(f"source numpy array: {np_array1}")
    print(f"convert to tensor: {tensor1}")

    # change the np array value reflects on tensor
    np.add(np_array1, 5, out=np_array1)
    print(f"tensor value after np.add(5): {tensor1}")
    print(f"np array value after np.add(5): {np_array1}")


def main():
    # tensor_creation()

    # tensor_attributes()

    # tensor_operations()

    tensor_np_conversion()


if __name__ == "__main__":
    main()
