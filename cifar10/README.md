ConvNetGPU.py :  A simple convolutional Neural Network on cifar10 data. Ran for 200 epochs on a Tesla K80 GPU. Accuracy achieved 82.48%.

![Alt text](https://github.com/Shobhim/Convolutional-Neural-Networks/blob/master/cifar10/convNetGPU.png)

AllCNN.py : Implementation of the network defined in the striving for simplicity paper. Accuracy achieved 92.12%.

Changes that I made :

- Removed the first dropout layer.
- Reduced the weight decay in SGD from 1e-3 to 1e-6.

![Alt text](https://github.com/Shobhim/Convolutional-Neural-Networks/blob/master/cifar10/AllCNN_accuracy.png)

![Alt text](https://github.com/Shobhim/Convolutional-Neural-Networks/blob/master/cifar10/AllCNN_loss.png)
