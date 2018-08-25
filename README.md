# AI-ML-DL-02-Intro-to-TensorFlow
Udacity Self-Driving Car Engineer Nanodegree: Introduction to TensorFlow

## Linear Function

```python
def get_weights(n_features, n_labels):
    """
    Return TensorFlow weights
    :param n_features: Number of features
    :param n_labels: Number of labels
    :return: TensorFlow weights
    """
    return tf.Variable(tf.truncated_normal((n_features, n_labels)))
```
I'll use the ``tf.truncated_normal()`` function to generate random numbers from a normal distribution.
The ``tf.truncated_normal()`` function returns a tensor with random values from a normal distribution whose magnitude is no more than 2 standard deviations from the mean. 

```python
def get_biases(n_labels):
    """
    Return TensorFlow bias
    :param n_labels: Number of labels
    :return: TensorFlow bias
    """
    return tf.Variable(tf.zeros(n_labels))
```
Since the weights are already helping prevent the model from getting stuck, you don't need to randomize the bias.
The ``tf.zeros()`` function returns a tensor with all zeros.

```python
def linear(input, w, b):
    """
    Return linear function in TensorFlow
    :param input: TensorFlow input
    :param w: TensorFlow weights
    :param b: TensorFlow biases
    :return: TensorFlow linear function
    """
    return tf.add(tf.matmul(input, w), b)
```
Implement ``xW + b`` in the linear function.

## Cross Entropy

<img src="https://github.com/ChenBohan/AI-ML-DL-02-Intro-to-TensorFlow/blob/master/readme_img/cross-entropy.png" width = "50%" height = "50%" div align=center />

## SGD

### Momentum

### Learning rate decay
