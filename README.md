# Apple-recognition neural network

This neural network recognizes if an input image contains an apple or not. 

The parts of the code for the neural network come from https://github.com/mnielsen/neural-networks-and-deep-learning.
That repository accompanies Michael Nielson's online book 'Neural Networks and Deep Learning', in which the fundamentals of neural networks are explained.
I have modified his code at certain points.

The network consists of 9408 input neurons (representing a 56 x 56 px input image, where each pixel contains three RGB values), two hidden layers with 500 and 25 neurons, respectively, and 1 output neuron (representing the probability with which the network thinks the input image contains and apple). 

The network is quite standard; it trains through gradient descent (uses backpropogation to calculate the gradient vector of the cost function).
However, additional techniques were also used to accelerate the learning rate of the network and improve the initialization of its weights and biases.

* I created a centered, circular boundary with diameter = 6/8 * side length of the input image. For each pixel outside this boundary, I initialized every weight attached to that pixel to be 0.
  This technique relies on the fact that if an input image does contain an apple, that apple will most likely be centered. As such the background pixels surrounding the apple should have no effect on the output of the network.

The remaining techniques are outlined in the 3rd chapter of Nielson's book, 'Improving the way the neural networks learn'.
* Stochastic gradient descent trades off optimized gradient descent for faster training speeds.
* Cross-entropy cost function increases the learning rate of the final hidden layer, when compared to the quadratic cost function.
* Modified guassian distribution for the initialization of the weights decreases the number of saturated neurons, allowing for faster learning.
* Network has capability for L2 regularization.

I created the file data_collect.py to collect the images in the apple_data directory, resize them, store their RGB and luminance values in arrays, and feed that data to the neural network.

The network trained on a dataset of 1775 pictures (all of which were taken by me). I took 936 pictures of apples, and 839 picutres of non-apples.
I first trained the network for 50 epochs, with a learning rate of 0.5. When its accuracy with the training data surpassed a certain threshold (95% at first), I stored its specific weights and biases in parameters.json.
After that, I would initialize the network with the values in parameters.json before training it more (I also decreased the learning rate to 0.1). After several training sessions, the network's accuracy with the training data is now 97%.

I collected a small sample of new test images, and ran them through the network. Its accuracy was 100%.

![Image of test results](Test%20results.png)

This program is still not done yet, and I would like to add some front-end design to it in the future.
