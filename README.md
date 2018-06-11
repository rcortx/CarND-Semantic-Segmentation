# Semantic Segmentation Project
In this project, an advanced neural network is used to classify road vs non-road segments in images from [the Kitti Road dataset](http://www.cvlibs.net/download.php?file=data_road.zip).

## Project Info
To see the implementation please go to `main.py` in the CarND-Semantic-Segmentation repository.

## Overview
The goal of this project was to create a a fully convolutional neural network based on the VGG-16 image classifier architecture to perform semantic segmentation (inorder to segment road chunks).

## Architecture
Using a pre-trained VGG-16 network converted to a fully convolutional neural network via replacing the final FCC layer with a 1x1 convolution and setting the depth to the desired number of classes.

The use of skip connections, 1x1 convolutions on prior layers, and finally upsampling (to output the same size image as the input) further helped me achieve great results. Lastly, as recommended, I also implemented a kernal initializer and regularizer. These helped tremendously.

The main section of the code where this was all achieved can be seen below:

```
# 1x1 convolution of vgg layer 7
layer7_out = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='same', 
                            kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

# upsample
layer4a_in = tf.layers.conv2d_transpose(layer7_out, num_classes, 4, strides=(2, 2), padding='same', 
                            kernel_initializer= tf.random_normal_initializer(stddev=0.01), 
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

# 1x1 convolution of vgg layer 4
layer4b_in = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding='same', 
                            kernel_initializer= tf.random_normal_initializer(stddev=0.01), 
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

# skip connection
layer4a_out = tf.add(layer4a_in, layer4b_in)

# upsample
layer3a_in = tf.layers.conv2d_transpose(layer4a_out, num_classes, 4, strides=(2, 2), padding='same', 
                            kernel_initializer= tf.random_normal_initializer(stddev=0.01), 
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

# 1x1 convolution of vgg layer 3
layer3b_in = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding='same', 
                            kernel_initializer= tf.random_normal_initializer(stddev=0.01), 
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

# skip connection
layer3a_out = tf.add(layer3a_in, layer3b_in)

# upsample
nn_last_layer = tf.layers.conv2d_transpose(layer3a_out, num_classes, 16, strides=(8, 8), padding= 'same', 
                            kernel_initializer= tf.random_normal_initializer(stddev=0.01), 
                            kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
```

### Optimizer
Loss function: Cross-entropy
Optimizer: Adam optimizer

### Training
Hyperparameters:

|  Input          |    MSE   |
|  -----          |  ------- |
|  keep_prob      |  0.5     |
|  learning_rate  |  0.0009  |
|  epochs         |  25      |
|  batch_size     |  5       |


### Results
Loss dropped continuously over 25 epochs, going from an average of 0.86 in epoch 1 to an average of 0.06 in epoch 25. Check sample below:

![Sample 1](https://github.com/rbcorx/CarND-Semantic-Segmentation/blob/master/1.png?raw=true "Sample 1")
![Sample 2](https://github.com/rbcorx/CarND-Semantic-Segmentation/blob/master/2.png?raw=true "Sample 2")

# Semantic Segmentation
### Introduction
In this project, you'll label the pixels of a road in images using a Fully Convolutional Network (FCN).

### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder  (**all images from the most recent run**)
 
 ### Tips
- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip)
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [forum post](https://discussions.udacity.com/t/here-is-some-advice-and-clarifications-about-the-semantic-segmentation-project/403100/8?u=subodh.malgonde) for more information.  A summary of additional points, follow. 
- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy. 
- When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented.
 
### Using GitHub and Creating Effective READMEs
If you are unfamiliar with GitHub , Udacity has a brief [GitHub tutorial](http://blog.udacity.com/2015/06/a-beginners-git-github-tutorial.html) to get you started. Udacity also provides a more detailed free [course on git and GitHub](https://www.udacity.com/course/how-to-use-git-and-github--ud775).

To learn about REAMDE files and Markdown, Udacity provides a free [course on READMEs](https://www.udacity.com/courses/ud777), as well. 

GitHub also provides a [tutorial](https://guides.github.com/features/mastering-markdown/) about creating Markdown files.
