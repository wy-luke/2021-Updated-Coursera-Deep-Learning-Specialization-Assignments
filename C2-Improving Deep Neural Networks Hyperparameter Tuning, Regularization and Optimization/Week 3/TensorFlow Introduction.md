# Introduction to TensorFlow

Welcome to this week's programming assignment! Up until now, you've always used Numpy to build neural networks, but this week you'll explore a deep learning framework that allows you to build neural networks more easily. Machine learning frameworks like TensorFlow, PaddlePaddle, Torch, Caffe, Keras, and many others can speed up your machine learning development significantly. TensorFlow 2.3 has made significant improvements over its predecessor, some of which you'll encounter and implement here!

By the end of this assignment, you'll be able to do the following in TensorFlow 2.3:

* Use `tf.Variable` to modify the state of a variable
* Explain the difference between a variable and a constant
* Apply TensorFlow decorators to speed up code
* Train a Neural Network on a TensorFlow dataset

Programming frameworks like TensorFlow not only cut down on time spent coding, but can also perform optimizations that speed up the code itself. 

## Table of Contents
- [1- Packages](#1)
    - [1.1 - Checking TensorFlow Version](#1-1)
- [2 - Basic Optimization with GradientTape](#2)
    - [2.1 - Linear Function](#2-1)
        - [Exercise 1 - linear_function](#ex-1)
    - [2.2 - Computing the Sigmoid](#2-2)
        - [Exercise 2 - sigmoid](#ex-2)
    - [2.3 - Using One Hot Encodings](#2-3)
        - [Exercise 3 - one_hot_matrix](#ex-3)
    - [2.4 - Initialize the Parameters](#2-4)
        - [Exercise 4 - initialize_parameters](#ex-4)
- [3 - Building Your First Neural Network in TensorFlow](#3)
    - [3.1 - Implement Forward Propagation](#3-1)
        - [Exercise 5 - forward_propagation](#ex-5)
    - [3.2 Compute the Cost](#3-2)
        - [Exercise 6 - compute_cost](#ex-6)
    - [3.3 - Train the Model](#3-3)
- [4 - Bibliography](#4)

<a name='1'></a>
## 1 - Packages


```python
import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework.ops import EagerTensor
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
import time
```

<a name='1-1'></a>
### 1.1 - Checking TensorFlow Version 

You will be using v2.3 for this assignment, for maximum speed and efficiency.


```python
tf.__version__
```




    '2.3.0'



<a name='2'></a>
## 2 - Basic Optimization with GradientTape

The beauty of TensorFlow 2 is in its simplicity. Basically, all you need to do is implement forward propagation through a computational graph. TensorFlow will compute the derivatives for you, by moving backwards through the graph recorded with `GradientTape`. All that's left for you to do then is specify the cost function and optimizer you want to use! 

When writing a TensorFlow program, the main object to get used and transformed is the `tf.Tensor`. These tensors are the TensorFlow equivalent of Numpy arrays, i.e. multidimensional arrays of a given data type that also contain information about the computational graph.

Below, you'll use `tf.Variable` to store the state of your variables. Variables can only be created once as its initial value defines the variable shape and type. Additionally, the `dtype` arg in `tf.Variable` can be set to allow data to be converted to that type. But if none is specified, either the datatype will be kept if the initial value is a Tensor, or `convert_to_tensor` will decide. It's generally best for you to specify directly, so nothing breaks!


Here you'll call the TensorFlow dataset created on a HDF5 file, which you can use in place of a Numpy array to store your datasets. You can think of this as a TensorFlow data generator! 

You will use the Hand sign data set, that is composed of images with shape 64x64x3.


```python
train_dataset = h5py.File('datasets/train_signs.h5', "r")
test_dataset = h5py.File('datasets/test_signs.h5', "r")
```


```python
x_train = tf.data.Dataset.from_tensor_slices(train_dataset['train_set_x'])
y_train = tf.data.Dataset.from_tensor_slices(train_dataset['train_set_y'])

x_test = tf.data.Dataset.from_tensor_slices(test_dataset['test_set_x'])
y_test = tf.data.Dataset.from_tensor_slices(test_dataset['test_set_y'])
```


```python
type(x_train)
```




    tensorflow.python.data.ops.dataset_ops.TensorSliceDataset



Since TensorFlow Datasets are generators, you can't access directly the contents unless you iterate over them in a for loop, or by explicitly creating a Python iterator using `iter` and consuming its
elements using `next`. Also, you can inspect the `shape` and `dtype` of each element using the `element_spec` attribute.


```python
print(y_train.element_spec)
```

    TensorSpec(shape=(), dtype=tf.int64, name=None)



```python
print(next(iter(x_train)))
```

    tf.Tensor(
    [[[227 220 214]
      [227 221 215]
      [227 222 215]
      ...
      [232 230 224]
      [231 229 222]
      [230 229 221]]
    
     [[227 221 214]
      [227 221 215]
      [228 221 215]
      ...
      [232 230 224]
      [231 229 222]
      [231 229 221]]
    
     [[227 221 214]
      [227 221 214]
      [227 221 215]
      ...
      [232 230 224]
      [231 229 223]
      [230 229 221]]
    
     ...
    
     [[119  81  51]
      [124  85  55]
      [127  87  58]
      ...
      [210 211 211]
      [211 212 210]
      [210 211 210]]
    
     [[119  79  51]
      [124  84  55]
      [126  85  56]
      ...
      [210 211 210]
      [210 211 210]
      [209 210 209]]
    
     [[119  81  51]
      [123  83  55]
      [122  82  54]
      ...
      [209 210 210]
      [209 210 209]
      [208 209 209]]], shape=(64, 64, 3), dtype=uint8)



```python
for element in x_train:
    print(element)
    break
```

    tf.Tensor(
    [[[227 220 214]
      [227 221 215]
      [227 222 215]
      ...
      [232 230 224]
      [231 229 222]
      [230 229 221]]
    
     [[227 221 214]
      [227 221 215]
      [228 221 215]
      ...
      [232 230 224]
      [231 229 222]
      [231 229 221]]
    
     [[227 221 214]
      [227 221 214]
      [227 221 215]
      ...
      [232 230 224]
      [231 229 223]
      [230 229 221]]
    
     ...
    
     [[119  81  51]
      [124  85  55]
      [127  87  58]
      ...
      [210 211 211]
      [211 212 210]
      [210 211 210]]
    
     [[119  79  51]
      [124  84  55]
      [126  85  56]
      ...
      [210 211 210]
      [210 211 210]
      [209 210 209]]
    
     [[119  81  51]
      [123  83  55]
      [122  82  54]
      ...
      [209 210 210]
      [209 210 209]
      [208 209 209]]], shape=(64, 64, 3), dtype=uint8)


There's one more additional difference between TensorFlow datasets and Numpy arrays: If you need to transform one, you would invoke the `map` method to apply the function passed as an argument to each of the elements.


```python
def normalize(image):
    """
    Transform an image into a tensor of shape (64 * 64 * 3, 1)
    and normalize its components.
    
    Arguments
    image - Tensor.
    
    Returns: 
    result -- Transformed tensor 
    """
    image = tf.cast(image, tf.float32) / 256.0
    image = tf.reshape(image, [-1,1])
    return image
```


```python
new_train = x_train.map(normalize)
new_test = x_test.map(normalize)
```


```python
new_train.element_spec
```




    TensorSpec(shape=(12288, 1), dtype=tf.float32, name=None)




```python
print(next(iter(new_train)))
```

    tf.Tensor(
    [[0.88671875]
     [0.859375  ]
     [0.8359375 ]
     ...
     [0.8125    ]
     [0.81640625]
     [0.81640625]], shape=(12288, 1), dtype=float32)


<a name='2-1'></a>
### 2.1 - Linear Function

Let's begin this programming exercise by computing the following equation: $Y = WX + b$, where $W$ and $X$ are random matrices and b is a random vector. 

<a name='ex-1'></a>
### Exercise 1 - linear_function

Compute $WX + b$ where $W, X$, and $b$ are drawn from a random normal distribution. W is of shape (4, 3), X is (3,1) and b is (4,1). As an example, this is how to define a constant X with the shape (3,1):
```python
X = tf.constant(np.random.randn(3,1), name = "X")

```
Note that the difference between `tf.constant` and `tf.Variable` is that you can modify the state of a `tf.Variable` but cannot change the state of a `tf.constant`.

You might find the following functions helpful: 
- tf.matmul(..., ...) to do a matrix multiplication
- tf.add(..., ...) to do an addition
- np.random.randn(...) to initialize randomly


```python
# GRADED FUNCTION: linear_function

def linear_function():
    """
    Implements a linear function: 
            Initializes X to be a random tensor of shape (3,1)
            Initializes W to be a random tensor of shape (4,3)
            Initializes b to be a random tensor of shape (4,1)
    Returns: 
    result -- Y = WX + b 
    """

    np.random.seed(1)
    
    """
    Note, to ensure that the "random" numbers generated match the expected results,
    please create the variables in the order given in the starting code below.
    (Do not re-arrange the order).
    """
    # (approx. 4 lines)
    # X = ...
    # W = ...
    # b = ...
    # Y = ...
    # YOUR CODE STARTS HERE
    X = tf.constant(np.random.randn(3,1), name = "X")
    W = tf.constant(np.random.randn(4,3), name = "W")
    b = tf.constant(np.random.randn(4,1), name = "b")
    Y = tf.add(tf.matmul(W, X), b)
    
    # YOUR CODE ENDS HERE
    return Y
```


```python
result = linear_function()
print(result)

assert type(result) == EagerTensor, "Use the TensorFlow API"
assert np.allclose(result, [[-2.15657382], [ 2.95891446], [-1.08926781], [-0.84538042]]), "Error"
print("\033[92mAll test passed")

```

    tf.Tensor(
    [[-2.15657382]
     [ 2.95891446]
     [-1.08926781]
     [-0.84538042]], shape=(4, 1), dtype=float64)
    [92mAll test passed


**Expected Output**: 

```
result = 
[[-2.15657382]
 [ 2.95891446]
 [-1.08926781]
 [-0.84538042]]
```

<a name='2-2'></a>
### 2.2 - Computing the Sigmoid 
Amazing! You just implemented a linear function. TensorFlow offers a variety of commonly used neural network functions like `tf.sigmoid` and `tf.softmax`.

For this exercise, compute the sigmoid of z. 

In this exercise, you will: Cast your tensor to type `float32` using `tf.cast`, then compute the sigmoid using `tf.keras.activations.sigmoid`. 

<a name='ex-2'></a>
### Exercise 2 - sigmoid

Implement the sigmoid function below. You should use the following: 

- `tf.cast("...", tf.float32)`
- `tf.keras.activations.sigmoid("...")`


```python
# GRADED FUNCTION: sigmoid

def sigmoid(z):
    
    """
    Computes the sigmoid of z
    
    Arguments:
    z -- input value, scalar or vector
    
    Returns: 
    a -- (tf.float32) the sigmoid of z
    """
    # tf.keras.activations.sigmoid requires float16, float32, float64, complex64, or complex128.
    
    # (approx. 2 lines)
    # z = tf.cast("...", tf.float32)
    # a = ...
    # YOUR CODE STARTS HERE
    z = tf.cast(z, tf.float32)
    a = tf.keras.activations.sigmoid(z)
    # YOUR CODE ENDS HERE
    return a

```


```python
result = sigmoid(-1)
print ("type: " + str(type(result)))
print ("dtype: " + str(result.dtype))
print ("sigmoid(-1) = " + str(result))
print ("sigmoid(0) = " + str(sigmoid(0.0)))
print ("sigmoid(12) = " + str(sigmoid(12)))

def sigmoid_test(target):
    result = target(0)
    assert(type(result) == EagerTensor)
    assert (result.dtype == tf.float32)
    assert sigmoid(0) == 0.5, "Error"
    assert sigmoid(-1) == 0.26894143, "Error"
    assert sigmoid(12) == 0.9999939, "Error"

    print("\033[92mAll test passed")

sigmoid_test(sigmoid)
```

    type: <class 'tensorflow.python.framework.ops.EagerTensor'>
    dtype: <dtype: 'float32'>
    sigmoid(-1) = tf.Tensor(0.26894143, shape=(), dtype=float32)
    sigmoid(0) = tf.Tensor(0.5, shape=(), dtype=float32)
    sigmoid(12) = tf.Tensor(0.9999939, shape=(), dtype=float32)
    [92mAll test passed


**Expected Output**: 
<table>
<tr> 
<td>
type
</td>
<td>
class 'tensorflow.python.framework.ops.EagerTensor'
</td>
</tr><tr> 
<td>
dtype
</td>
<td>
"dtype: 'float32'
</td>
</tr>
<tr> 
<td>
Sigmoid(-1)
</td>
<td>
0.2689414
</td>
</tr>
<tr> 
<td>
Sigmoid(0)
</td>
<td>
0.5
</td>
</tr>
<tr> 
<td>
Sigmoid(12)
</td>
<td>
0.999994
</td>
</tr> 

</table> 

<a name='2-3'></a>
### 2.3 - Using One Hot Encodings

Many times in deep learning you will have a $Y$ vector with numbers ranging from $0$ to $C-1$, where $C$ is the number of classes. If $C$ is for example 4, then you might have the following y vector which you will need to convert like this:


<img src="images/onehot.png" style="width:600px;height:150px;">

This is called "one hot" encoding, because in the converted representation, exactly one element of each column is "hot" (meaning set to 1). To do this conversion in numpy, you might have to write a few lines of code. In TensorFlow, you can use one line of code: 

- [tf.one_hot(labels, depth, axis=0)](https://www.tensorflow.org/api_docs/python/tf/one_hot)

`axis=0` indicates the new axis is created at dimension 0

<a name='ex-3'></a>
### Exercise 3 - one_hot_matrix

Implement the function below to take one label and the total number of classes $C$, and return the one hot encoding in a column wise matrix. Use `tf.one_hot()` to do this, and `tf.reshape()` to reshape your one hot tensor! 

- `tf.reshape(tensor, shape)`


```python
# GRADED FUNCTION: one_hot_matrix
def one_hot_matrix(label, depth=6):
    """
    Computes the one hot encoding for a single label
    
    Arguments:
        label --  (int) Categorical labels
        depth --  (int) Number of different classes that label can take
    
    Returns:
         one_hot -- tf.Tensor A single-column matrix with the one hot encoding.
    """
    # (approx. 1 line)
    # one_hot = ...
    # YOUR CODE STARTS HERE
    one_hot = tf.reshape(tf.one_hot(label, depth, axis=0), [depth, 1])
    
    # YOUR CODE ENDS HERE
    return one_hot
```


```python
def one_hot_matrix_test(target):
    label = tf.constant(1)
    depth = 4
    result = target(label, depth)
    print(result)
    assert result.shape[0] == depth, "Use the parameter depth"
    assert result.shape[1] == 1, f"Reshape to have only 1 column"
    assert np.allclose(result,  [[0.], [1.], [0.], [0.]] ), "Wrong output. Use tf.one_hot"
    result = target(3, depth)
    assert np.allclose(result, [[0.], [0.], [0.], [1.]] ), "Wrong output. Use tf.one_hot"
    
    print("\033[92mAll test passed")

one_hot_matrix_test(one_hot_matrix)
```

    tf.Tensor(
    [[0.]
     [1.]
     [0.]
     [0.]], shape=(4, 1), dtype=float32)
    [92mAll test passed


**Expected output**
```
tf.Tensor(
[[0.]
 [1.]
 [0.]
 [0.]], shape=(4, 1), dtype=float32)
```


```python
new_y_test = y_test.map(one_hot_matrix)
new_y_train = y_train.map(one_hot_matrix)
```


```python
print(next(iter(new_y_test)))
```

    tf.Tensor(
    [[1.]
     [0.]
     [0.]
     [0.]
     [0.]
     [0.]], shape=(6, 1), dtype=float32)


<a name='2-4'></a>
### 2.4 - Initialize the Parameters 

Now you'll initialize a vector of numbers between zero and one. The function you'll be calling is `tf.keras.initializers.GlorotNormal`, which draws samples from a truncated normal distribution centered on 0, with `stddev = sqrt(2 / (fan_in + fan_out))`, where `fan_in` is the number of input units and `fan_out` is the number of output units, both in the weight tensor. 

To initialize with zeros or ones you could use `tf.zeros()` or `tf.ones()` instead. 

<a name='ex-4'></a>
### Exercise 4 - initialize_parameters

Implement the function below to take in a shape and to return an array of numbers between -1 and 1. 

 - `tf.keras.initializers.GlorotNormal(seed=1)`
 - `tf.Variable(initializer(shape=())`


```python
# GRADED FUNCTION: initialize_parameters

def initialize_parameters():
    """
    Initializes parameters to build a neural network with TensorFlow. The shapes are:
                        W1 : [25, 12288]
                        b1 : [25, 1]
                        W2 : [12, 25]
                        b2 : [12, 1]
                        W3 : [6, 12]
                        b3 : [6, 1]
    
    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """
                                
    initializer = tf.keras.initializers.GlorotNormal(seed=1)
    #(approx. 6 lines of code)
    # W1 = ...
    # b1 = ...
    # W2 = ...
    # b2 = ...
    # W3 = ...
    # b3 = ...
    # YOUR CODE STARTS HERE
    W1 = tf.Variable(initializer(shape=([25, 12288])))
    b1 = tf.Variable(initializer(shape=([25, 1])))
    W2 = tf.Variable(initializer(shape=([12, 25])))
    b2 = tf.Variable(initializer(shape=([12, 1])))
    W3 = tf.Variable(initializer(shape=([6, 12])))
    b3 = tf.Variable(initializer(shape=([6, 1])))
    
    # YOUR CODE ENDS HERE

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return parameters
```


```python
def initialize_parameters_test(target):
    parameters = target()

    values = {"W1": (25, 12288),
              "b1": (25, 1),
              "W2": (12, 25),
              "b2": (12, 1),
              "W3": (6, 12),
              "b3": (6, 1)}

    for key in parameters:
        print(f"{key} shape: {tuple(parameters[key].shape)}")
        assert type(parameters[key]) == ResourceVariable, "All parameter must be created using tf.Variable"
        assert tuple(parameters[key].shape) == values[key], f"{key}: wrong shape"
        assert np.abs(np.mean(parameters[key].numpy())) < 0.5,  f"{key}: Use the GlorotNormal initializer"
        assert np.std(parameters[key].numpy()) > 0 and np.std(parameters[key].numpy()) < 1, f"{key}: Use the GlorotNormal initializer"

    print("\033[92mAll test passed")
    
initialize_parameters_test(initialize_parameters)
```

    W1 shape: (25, 12288)
    b1 shape: (25, 1)
    W2 shape: (12, 25)
    b2 shape: (12, 1)
    W3 shape: (6, 12)
    b3 shape: (6, 1)
    [92mAll test passed


**Expected output**
```
W1 shape: (25, 12288)
b1 shape: (25, 1)
W2 shape: (12, 25)
b2 shape: (12, 1)
W3 shape: (6, 12)
b3 shape: (6, 1)
```


```python
parameters = initialize_parameters()
```

<a name='3'></a>
## 3 - Building Your First Neural Network in TensorFlow

In this part of the assignment you will build a neural network using TensorFlow. Remember that there are two parts to implementing a TensorFlow model:

- Implement forward propagation
- Retrieve the gradients and train the model

Let's get into it!

<a name='3-1'></a>
### 3.1 - Implement Forward Propagation 

One of TensorFlow's great strengths lies in the fact that you only need to implement the forward propagation function. 

Here, you'll use a TensorFlow decorator, `@tf.function`, which builds a  computational graph to execute the function. `@tf.function` is polymorphic, which comes in very handy, as it can support arguments with different data types or shapes, and be used with other languages, such as Python. This means that you can use data dependent control flow statements.

When you use `@tf.function` to implement forward propagation, the computational graph is activated, which keeps track of the operations. This is so you can calculate your gradients with backpropagation.

<a name='ex-5'></a>
### Exercise 5 - forward_propagation

Implement the `forward_propagation` function.

**Note** Use only the TF API. 

- tf.math.add
- tf.linalg.matmul
- tf.keras.activations.relu



```python
# GRADED FUNCTION: forward_propagation

@tf.function
def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    #(approx. 5 lines)                   # Numpy Equivalents:
    # Z1 = ...                           # Z1 = np.dot(W1, X) + b1
    # A1 = ...                           # A1 = relu(Z1)
    # Z2 = ...                           # Z2 = np.dot(W2, A1) + b2
    # A2 = ...                           # A2 = relu(Z2)
    # Z3 = ...                           # Z3 = np.dot(W3, A2) + b3
    # YOUR CODE STARTS HERE
    Z1 = tf.math.add(tf.linalg.matmul(W1, X), b1)
    A1 = tf.keras.activations.relu(Z1)
    Z2 = tf.math.add(tf.linalg.matmul(W2, A1), b2)
    A2 = tf.keras.activations.relu(Z2)
    Z3 = tf.math.add(tf.linalg.matmul(W3, A2), b3)
    
    # YOUR CODE ENDS HERE
    
    return Z3
```


```python
def forward_propagation_test(target, examples):
    for batch in examples:
        forward_pass = target(batch, parameters)
        assert type(forward_pass) == EagerTensor, "Your output is not a tensor"
        assert forward_pass.shape == (6, 1), "Last layer must use W3 and b3"
        assert np.any(forward_pass < 0), "Don't use a ReLu layer at end of your network"
        assert np.allclose(forward_pass, 
                           [[-0.13082162],
                           [ 0.21228778],
                           [ 0.7050022 ],
                           [-1.1224034 ],
                           [-0.20386729],
                           [ 0.9526217 ]]), "Output does not match"
        print(forward_pass)
        break
    

    print("\033[92mAll test passed")

forward_propagation_test(forward_propagation, new_train)
```

    tf.Tensor(
    [[-0.13082162]
     [ 0.21228778]
     [ 0.7050022 ]
     [-1.1224034 ]
     [-0.20386729]
     [ 0.9526217 ]], shape=(6, 1), dtype=float32)
    [92mAll test passed


**Expected output**
```
tf.Tensor(
[[-0.13082162]
 [ 0.21228778]
 [ 0.7050022 ]
 [-1.1224034 ]
 [-0.20386732]
 [ 0.9526217 ]], shape=(6, 1), dtype=float32)
```

<a name='3-2'></a>
### 3.2 Compute the Cost

Here again, the delightful `@tf.function` decorator steps in and saves you time. All you need to do is specify how to compute the cost, and you can do so in one simple step by using:

`tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true = ..., y_pred = ..., from_logits=True))`

<a name='ex-6'></a>
### Exercise 6 -  compute_cost

Implement the cost function below. 
- It's important to note that the "`y_pred`" and "`y_true`" inputs of [tf.keras.losses.binary_crossentropy](https://www.tensorflow.org/api_docs/python/tf/keras/losses/binary_crossentropy) are expected to be of shape (number of examples, num_classes). Since both the transpose and the original tensors have the same values, just in different order, the result of calculating the binary_crossentropy should be the same if you transpose or not the logits and labels. Just for reference here is how the Binary Cross entropy is calculated in TensorFlow:

``mean_reduce(max(logits, 0) - logits * labels + log(1 + exp(-abs(logits))), axis=-1)``

- `tf.reduce_mean` basically does the summation over the examples.


```python
# GRADED FUNCTION: compute_cost 

@tf.function
def compute_cost(logits, labels):
    """
    Computes the cost
    
    Arguments:
    logits -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    labels -- "true" labels vector, same shape as Z3
    
    Returns:
    cost - Tensor of the cost function
    """
    
    #(1 line of code)
    # cost = ...
    # YOUR CODE STARTS HERE
    cost = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true = labels, y_pred = logits, from_logits=True))
    
    # YOUR CODE ENDS HERE
    return cost
```


```python
def compute_cost_test(target):
    labels = np.array([[0., 1.], [0., 0.], [1., 0.]])
    logits = np.array([[0.6, 0.4], [0.4, 0.6], [0.4, 0.6]])
    result = compute_cost(logits, labels)
    print(result)
    assert(type(result) == EagerTensor), "Use the TensorFlow API"
    assert (np.abs(result - (0.7752516 +  0.9752516 + 0.7752516) / 3.0) < 1e-7), "Test does not match. Did you get the mean of your cost functions?"

    print("\033[92mAll test passed")

compute_cost_test(compute_cost)
```

    tf.Tensor(0.8419182681095858, shape=(), dtype=float64)
    [92mAll test passed


**Expected output**
```
tf.Tensor(0.8419182681095858, shape=(), dtype=float64)
```

<a name='3-3'></a>
### 3.3 - Train the Model

Let's talk optimizers. You'll specify the type of optimizer in one line, in this case `tf.keras.optimizers.Adam` (though you can use others such as SGD), and then call it within the training loop. 

Notice the `tape.gradient` function: this allows you to retrieve the operations recorded for automatic differentiation inside the `GradientTape` block. Then, calling the optimizer method `apply_gradients`, will apply the optimizer's update rules to each trainable parameter. At the end of this assignment, you'll find some documentation that explains this more in detail, but for now, a simple explanation will do. ;) 


Here you should take note of an important extra step that's been added to the batch training process: 

- `tf.Data.dataset = dataset.prefetch(8)` 

What this does is prevent a memory bottleneck that can occur when reading from disk. `prefetch()` sets aside some data and keeps it ready for when it's needed. It does this by creating a source dataset from your input data, applying a transformation to preprocess the data, then iterating over the dataset the specified number of elements at a time. This works because the iteration is streaming, so the data doesn't need to fit into the memory. 


```python
def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001,
          num_epochs = 1500, minibatch_size = 32, print_cost = True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
    
    Arguments:
    X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
    Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
    X_test -- training set, of shape (input size = 12288, number of training examples = 120)
    Y_test -- test set, of shape (output size = 6, number of test examples = 120)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    costs = []                                        # To keep track of the cost
    
    # Initialize your parameters
    #(1 line)
    parameters = initialize_parameters()

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    optimizer = tf.keras.optimizers.SGD(learning_rate)

    X_train = X_train.batch(minibatch_size, drop_remainder=True).prefetch(8)# <<< extra step    
    Y_train = Y_train.batch(minibatch_size, drop_remainder=True).prefetch(8) # loads memory faster 

    # Do the training loop
    for epoch in range(num_epochs):

        epoch_cost = 0.
        
        for (minibatch_X, minibatch_Y) in zip(X_train, Y_train):
            # Select a minibatch
            with tf.GradientTape() as tape:
                # 1. predict
                Z3 = forward_propagation(minibatch_X, parameters)
                # 2. loss
                minibatch_cost = compute_cost(Z3, minibatch_Y)
                
            trainable_variables = [W1, b1, W2, b2, W3, b3]
            grads = tape.gradient(minibatch_cost, trainable_variables)
            optimizer.apply_gradients(zip(grads, trainable_variables))
            epoch_cost += minibatch_cost / minibatch_size

        # Print the cost every epoch
        if print_cost == True and epoch % 10 == 0:
            print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
        if print_cost == True and epoch % 5 == 0:
            costs.append(epoch_cost)

    # Plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per fives)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    # Save the parameters in a variable
    print ("Parameters have been trained!")

    return parameters
```


```python
model(new_train, new_y_train, new_test, new_y_test, num_epochs=200)
```

    Cost after epoch 0: 0.742591
    Cost after epoch 10: 0.614557
    Cost after epoch 20: 0.598900
    Cost after epoch 30: 0.588907
    Cost after epoch 40: 0.579898
    Cost after epoch 50: 0.570628
    Cost after epoch 60: 0.560898
    Cost after epoch 70: 0.550808
    Cost after epoch 80: 0.540497
    Cost after epoch 90: 0.488142
    Cost after epoch 100: 0.478271
    Cost after epoch 110: 0.472863
    Cost after epoch 120: 0.468990
    Cost after epoch 130: 0.466014
    Cost after epoch 140: 0.463661
    Cost after epoch 150: 0.461677
    Cost after epoch 160: 0.459951
    Cost after epoch 170: 0.458391
    Cost after epoch 180: 0.456969
    Cost after epoch 190: 0.455648



![png](output_50_1.png)


    Parameters have been trained!





    {'W1': <tf.Variable 'Variable:0' shape=(25, 12288) dtype=float32, numpy=
     array([[ 0.00159527, -0.00737913,  0.00893296, ..., -0.01227794,
              0.01642206,  0.00506491],
            [ 0.02264025,  0.0067227 ,  0.00795862, ...,  0.00284724,
              0.01910819,  0.00122853],
            [-0.00173585, -0.00872453, -0.01410444, ..., -0.00733837,
              0.0205085 , -0.02683027],
            ...,
            [-0.00126929,  0.01729332,  0.02082342, ...,  0.01709594,
              0.00429358, -0.00733263],
            [ 0.00268262,  0.004105  ,  0.00936713, ...,  0.01222287,
             -0.02717604,  0.01498359],
            [-0.00145541,  0.02459595,  0.00339064, ..., -0.02478788,
              0.02716016, -0.00306428]], dtype=float32)>,
     'b1': <tf.Variable 'Variable:0' shape=(25, 1) dtype=float32, numpy=
     array([[ 0.03964256],
            [-0.15545043],
            [ 0.19885883],
            [-0.24874453],
            [-0.2867676 ],
            [-0.12604603],
            [-0.01213098],
            [ 0.14784063],
            [-0.00413172],
            [-0.4408977 ],
            [ 0.54054177],
            [-0.4345032 ],
            [ 0.11763882],
            [ 0.21523887],
            [-0.06772587],
            [-0.16429274],
            [-0.05259617],
            [-0.18479495],
            [-0.00280256],
            [-0.06777475],
            [ 0.09226809],
            [ 0.02067652],
            [-0.05682073],
            [ 0.37065902],
            [ 0.21586621]], dtype=float32)>,
     'W2': <tf.Variable 'Variable:0' shape=(12, 25) dtype=float32, numpy=
     array([[ 0.03270398, -0.13031   ,  0.16566682, -0.20850259, -0.2404858 ,
             -0.10598166, -0.01016674,  0.12317107, -0.00411659, -0.3709333 ,
              0.45312327, -0.36423257,  0.09766971,  0.18042907, -0.05753209,
             -0.13796303, -0.04518652, -0.15597364, -0.00236228, -0.05681378,
              0.07734591,  0.01733258, -0.04763132,  0.31054643,  0.18095495],
            [ 0.275006  ,  0.0652916 ,  0.19277105,  0.00808901, -0.35061046,
             -0.04379591,  0.00529772,  0.14074473, -0.22700697, -0.08254652,
             -0.10437229, -0.27877635, -0.22737731, -0.15467171, -0.30434608,
              0.42841426,  0.04013019,  0.14082581,  0.40803406,  0.19127996,
             -0.08289494,  0.19833343, -0.18854785,  0.11045384, -0.10293514],
            [ 0.07370555,  0.12879197, -0.38048682, -0.1428371 , -0.16866712,
             -0.12560502,  0.08047906, -0.1422226 , -0.3291437 ,  0.11487076,
              0.21897362,  0.1428981 ,  0.4108547 , -0.02966296, -0.11487766,
              0.2835236 ,  0.2571582 , -0.1236593 ,  0.1469501 , -0.39992067,
             -0.11544652, -0.11918075, -0.5031594 , -0.16647008, -0.0463655 ],
            [-0.11886349,  0.19529893, -0.13205278, -0.46206364,  0.07806116,
             -0.36992028, -0.06379852,  0.37158278,  0.0755597 ,  0.5198782 ,
             -0.01714175,  0.35476214,  0.09361284,  0.17954443,  0.00514452,
              0.04280831,  0.1051797 ,  0.03766964, -0.23309673, -0.23678231,
             -0.07444265, -0.30713868, -0.11694647,  0.3292588 , -0.09511968],
            [ 0.1594041 ,  0.0393942 ,  0.47869283,  0.2265753 ,  0.03725046,
             -0.51921755, -0.01731534, -0.31578013, -0.21672064,  0.04122872,
              0.04947535, -0.29094276, -0.03152779,  0.47902155,  0.31676546,
              0.0473902 ,  0.07770423,  0.3139462 , -0.02500637,  0.10048122,
             -0.05332499, -0.34107792, -0.13928485,  0.12402116, -0.41300818],
            [-0.14994699,  0.03965309, -0.47870195, -0.07975383,  0.09755022,
             -0.00232862, -0.26367775, -0.23967475,  0.24946521,  0.22969191,
             -0.30773658,  0.1017215 ,  0.03053034,  0.26468748, -0.51858497,
             -0.08669744,  0.03128893,  0.28504866,  0.20724739, -0.14461054,
             -0.09631125,  0.2553377 ,  0.0313108 ,  0.28684464,  0.02228327],
            [-0.20329641, -0.2922766 , -0.03024991,  0.00603078,  0.34428513,
              0.14932795, -0.42723438,  0.07875892,  0.06157893, -0.19437575,
              0.03054013, -0.20949648,  0.2890019 ,  0.03168807,  0.18291238,
             -0.17629069, -0.2162296 ,  0.02522451, -0.17976451,  0.20999093,
              0.13074148,  0.12900151, -0.29620144,  0.39828372,  0.35581756],
            [-0.08132942,  0.0508789 ,  0.03970909, -0.06884057, -0.07758211,
              0.21220328,  0.16169944, -0.05766107, -0.04837854, -0.23052695,
              0.2551639 , -0.2933403 , -0.16104451, -0.11232601, -0.1305835 ,
              0.0502181 ,  0.18621859, -0.07786819,  0.10281896, -0.06372993,
              0.41251048, -0.01803587,  0.04746069,  0.27628538, -0.21901166],
            [ 0.28539097,  0.20629272, -0.38372156,  0.26297212,  0.2350495 ,
              0.18105377,  0.25501856, -0.19114897,  0.355807  ,  0.00106926,
             -0.33252424, -0.09722907, -0.00984821,  0.22310142, -0.22939995,
             -0.027319  ,  0.18572639, -0.00867896,  0.47467512,  0.00131025,
              0.3148377 , -0.22662118,  0.12927507,  0.04265415, -0.45121887],
            [-0.23054187, -0.22334962, -0.18913192,  0.15417175, -0.07368277,
             -0.0554374 ,  0.12214173,  0.3880139 , -0.01242276,  0.11768965,
              0.26777858, -0.06251994, -0.12100054, -0.12495217, -0.03189994,
             -0.50085783, -0.09560107, -0.2402923 ,  0.07087833,  0.03642716,
             -0.00494978, -0.36984688,  0.00878784,  0.24595837, -0.1323934 ],
            [ 0.3191285 ,  0.02266271, -0.06669848, -0.33996752,  0.36436087,
             -0.2986556 , -0.0511701 , -0.37243623,  0.27359596,  0.20692123,
              0.02171116,  0.10230298, -0.3980014 ,  0.02363082,  0.13089406,
              0.3354062 ,  0.08214818,  0.20031574, -0.081278  , -0.28784147,
              0.17327178, -0.1326688 ,  0.28894275, -0.19869894, -0.03405774],
            [ 0.18820512, -0.20398362, -0.03503615, -0.36792815, -0.22963929,
              0.23911732, -0.04237934, -0.0165515 , -0.05906188,  0.16423857,
             -0.32017106,  0.15379827,  0.148428  , -0.24647985, -0.08833542,
              0.13306345,  0.41101247,  0.36263198,  0.3355143 ,  0.05405051,
              0.21186371,  0.0197499 ,  0.45979315,  0.04402945,  0.36662805]],
           dtype=float32)>,
     'b2': <tf.Variable 'Variable:0' shape=(12, 1) dtype=float32, numpy=
     array([[ 0.05586328],
            [-0.22080165],
            [ 0.28016472],
            [-0.35078037],
            [-0.4071067 ],
            [-0.17678553],
            [-0.01738933],
            [ 0.20840582],
            [-0.00978936],
            [-0.6255955 ],
            [ 0.76646936],
            [-0.6127309 ]], dtype=float32)>,
     'W3': <tf.Variable 'Variable:0' shape=(6, 12) dtype=float32, numpy=
     array([[ 0.04761663, -0.1869142 ,  0.23871909, -0.29910955, -0.34814283,
             -0.14891681, -0.01468904,  0.17718893, -0.00528874, -0.53165394,
              0.6512269 , -0.52562165],
            [ 0.14137168,  0.2585655 , -0.08242776, -0.24897616, -0.08694383,
             -0.23123945, -0.00362752, -0.08165746,  0.10867161,  0.02485008,
             -0.10594384,  0.40712833],
            [ 0.2588232 ,  0.39531493,  0.09260565,  0.19811322, -0.02675355,
             -0.5141325 , -0.06277467,  0.0073774 ,  0.20107801, -0.3234572 ,
             -0.17168574, -0.21085034],
            [-0.39999744, -0.32459378, -0.22211906, -0.44647554,  0.60610884,
              0.06233876,  0.20539553,  0.5849541 ,  0.27438077, -0.11884821,
              0.27649078, -0.2822457 ],
            [ 0.15867612, -0.14753199,  0.10718007,  0.20221217, -0.5397337 ,
             -0.1961394 , -0.24190293, -0.17912963,  0.11743002, -0.20150405,
             -0.45716155,  0.17953752],
            [ 0.313728  ,  0.20475906,  0.5910115 , -0.07041359, -0.17219503,
              0.39685282,  0.37256533, -0.17371103,  0.2085042 , -0.5733746 ,
             -0.18640897, -0.19104898]], dtype=float32)>,
     'b3': <tf.Variable 'Variable:0' shape=(6, 1) dtype=float32, numpy=
     array([[ 0.07464746],
            [-0.31648052],
            [ 0.3577738 ],
            [-0.4854469 ],
            [-0.5509957 ],
            [-0.24964447]], dtype=float32)>}



**Expected output**

```
Cost after epoch 0: 0.742591
Cost after epoch 10: 0.614557
Cost after epoch 20: 0.598900
Cost after epoch 30: 0.588907
Cost after epoch 40: 0.579898
...
```

**Congratulations**! You've made it to the end of this assignment, and to the end of this week's material. Amazing work building a neural network in TensorFlow 2.3! 

Here's a quick recap of all you just achieved:

- Used `tf.Variable` to modify your variables
- Applied TensorFlow decorators and observed how they sped up your code
- Trained a Neural Network on a TensorFlow dataset

You are now able to harness the power of TensorFlow's computational graph to create cool things, faster. Nice! 

<a name='4'></a>
## 4 - Bibliography 

In this assignment, you were introducted to `tf.GradientTape`, which records operations for differentation. Here are a couple of resources for diving deeper into what it does and why: 

Introduction to Gradients and Automatic Differentiation: 
https://www.tensorflow.org/guide/autodiff 

GradientTape documentation:
https://www.tensorflow.org/api_docs/python/tf/GradientTape


```python

```
