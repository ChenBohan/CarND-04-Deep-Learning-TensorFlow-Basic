# AI-Deep-Learning-02-Intro-to-TensorFlow
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

<img src="https://github.com/ChenBohan/AI-ML-DL-02-Intro-to-TensorFlow/blob/master/readme_img/overview.png" width = "50%" height = "50%" div align=center />

<img src="https://github.com/ChenBohan/AI-ML-DL-02-Intro-to-TensorFlow/blob/master/readme_img/loss%20function.png" width = "50%" height = "50%" div align=center />

## SGD

### Mini-batching

The idea is to randomly shuffle the data at the start of each epoch, then create the mini-batches. 
For each mini-batch, you train the network weights with gradient descent. 

```python
features = tf.placeholder(tf.float32, [None, n_input])
labels = tf.placeholder(tf.float32, [None, n_classes])
```

In that case, the size of the batches would vary, so you need to take advantage of TensorFlow's ``tf.placeholder()`` function to receive the varying batch sizes.

The ``None`` dimension is a ``placeholder`` for the batch size. At runtime, TensorFlow will accept any batch size greater than 0.

Implement the batches function:

```python
def batches(batch_size, features, labels):
    """
    Create batches of features and labels
    :param batch_size: The batch size
    :param features: List of features
    :param labels: List of labels
    :return: Batches of (Features, Labels)
    """
    assert len(features) == len(labels)
    # TODO: Implement batching
    outout_batches = []    
    sample_size = len(features)
    for start_i in range(0, sample_size, batch_size):
        end_i = start_i + batch_size
        batch = [features[start_i:end_i], labels[start_i:end_i]]
        outout_batches.append(batch)       
    return outout_batches
```

Let's use mini-batching to feed batches of MNIST features and labels into a linear model.
```python
with tf.Session() as sess:
    sess.run(init)
    
    # TODO: Train optimizer on all batches
    for batch_features, batch_labels in batches(batch_size, train_features, train_labels):
        sess.run(optimizer, feed_dict={features: batch_features, labels: batch_labels})
        
    # Calculate accuracy for test dataset
    test_accuracy = sess.run(
        accuracy,
        feed_dict={features: test_features, labels: test_labels})
```

### Momentum 
<img src="https://github.com/ChenBohan/AI-ML-DL-02-Intro-to-TensorFlow/blob/master/readme_img/momentum.png" width = "50%" height = "50%" div align=center />

### Learning rate decay
<img src="https://github.com/ChenBohan/AI-ML-DL-02-Intro-to-TensorFlow/blob/master/readme_img/learning%20rate%20decay.png" width = "50%" height = "50%" div align=center />
