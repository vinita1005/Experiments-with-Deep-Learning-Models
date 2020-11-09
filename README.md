# Experiments-with-Deep-Learning-Models
There are many convolutional neural network architectures developed for various tasks such as classification and detection of images. In this project, 3 of such architectures are studied and implemented - VGGNet16, ResNet18 and Inception V2. Along with implementations of these architectures, various combinations of regularization techniques and optimizers are experimented. The regularization techniques used are - Batch Normalization and Dropout while optimizers include ADAM and SGD. The observations from these 18 experiments can help us analyse which optimizer works best with which regularization technique. Also, we will be able to identify how regularization helps in achieving higher accuracies than models with no regularization. These observations are discussed in detail in the results section of the report.
# Networks
1 VGGNet16
VGGNet16 is a cnn architecture proposed by K. Simonyan and A. Zisserman in the
paper "Very Deep Convolutional Networks for Large-Scale Image Recognition"[1]. It was
one of the most famous models used in image recognition and it even managed to
achieve 92.75% accuracy on the ImageNet dataset.
For the ImageNet dataset, they have taken a fixed input size of 224 x 224 RGB image.
For CIFAR100 dataset, the input size will be 32 x 32. The images will be passed
through several convolutional layers with different filter sizes: 2 layers of 64 filters, 3
layers of 128, 4 layers of 256 and then 4 layers of 521 filters twice. This shows that the
model is very complex and suited for large data-sets. Since we are using CIFAR100
dataset which is comparatively smaller than ImageNet, the model has been modified for
this project slightly.
The model is made up of 2 cnn layers of filter size 64, then 2 layers of 128 and 4 layers
of 256 filters. The discussion regarding why this architecture works for CIFAR100 is
mentioned in the results section of the report.
The final layers are- 2 fully connected layers of 4096 channels each and final softmax
layer which contains 100 channels for 100-classes of classification of CIFAR-100 dataset.

2 ResNet18
The Gradient vanishing problem has been the most troublesome problem in the history
of training deep neural network. Residual networks were introduced in hopes of dealing
with this problem.
The core idea of ResNet is introducing a so-called âAidentity shortcut connectionâAI
that skips one or more layers.[4]
The intuition was that, stacking deeper layers should not degrade the performance of
the neural network. Which means, the deeper model should perform better wrt to its
previous layer. There have been many advancements in Resnet architecture such as
ResNext etc.
In this project, a typical resnet architecture using 4 stacks of recidual blocks was
implemented.

3 InceptionNet V2
The inception network is more complex than a typical deep neural network. Inception
V2 was introduced [5] to improve the Inception architecture in terms of both speed and
accuracy. The intuition was to make the network more wider rather than deeper.
The figure shows that the inception block includes a layers with different filter sizes of
1x1, 3x3 and 5x5. The outputs of these layers are concatenated and sent to the next
inception block. [6]
In InceptionV2, to make deep neural networks cheaper, the architects of inception
network limited the number of input channels by adding 1x1 convolution layer before
the 3x3 and 5x5 layers.

# Regularizers
1 Early stopping
[9] Sometimes, while training on a large dataset, the model will stop learning and
instead focus on the external noise. To avoid this, early stopping is used in all the
experiments. This technique tells us just the right time to stop training of the model.
In this project, the early stopping is applied to observe saturation of validation loss and
patience is set to 20.

2 Batch Normalization
[8] Batch Normalization is a technique wherein the input layer is normalized by
adjusting and scaling the activations. It normalizes the output of the previous
activation layer by subtracting the mean and dividing by the batch standard deviation.
This ensures that the model has stable behaviour and avoids overfitting. The training
time is also reduced using this.

3 Dropout
[8] This method helps in reducing overfitting of the model. The nodes are removed
randomly trained network so that it does not overfit to the training data. So, at the
test time, we can approximate the average of all the predictions easily since the network
is already trained.The dropout rate in this project is specified to be 20%

# Variations of parameters used
1 Lower learning rates
In case of ADAM, the experiments with various learning rates showed that lower
learning rate such as "0.0001" yeilds a better solution. This also helps in overfitting
problem. Whearas in SGD, a higher learning rate such as "0.01" proved to be better.

2 Gradient clipping
[7] Gradient clipping is used to counter the problem of exploding gradients. Exploding
gradients occur when the gradients get too large in training period making the model
unstable while vanishing gradient means the gradients become too small for model to
learn anything. The idea of gradient clipping is ig the gradient gets too large, we clip it
to make it small. This helps the gradient descent to have a normal behaviour. Here,
gradient clipping is used with clipnorm=5.

3 Activation functions
The most used activation function ’relu’ is used in most of the experiments. But,
through multiple experiments it is seen that relu does not fit well with ADAM
optimizer. The model becomes very unstable and the loss functions varies a lot. Hence,
to tackle this unstable behaviour, ’elu’ and sometimes ’leakyrelu’ activations are used in
experiements considering ADAM optimizers.

# Results
The detailed analysis of the experiments with various hyperparameter tuning is provided in the results section of the report.
