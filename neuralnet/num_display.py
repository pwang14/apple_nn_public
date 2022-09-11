import network2
import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network2.Network([784, 30, 10])
net.SGD(list(training_data), 30, 10, 3.0, monitor_training_accuracy=True)
