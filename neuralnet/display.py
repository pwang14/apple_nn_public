import network2
import data_collect
import os

path = '..\\apple_data\\'
tr_data, tr_data_lu = data_collect.load_data(path+'training_data\\')
va_data, tr_data_lu = data_collect.load_data(path+'validation_data\\')

net = network2.load('parameters.json')
te_data, te_data_lu, te_data_l = data_collect.dir_load_data('../apple_data/test_data/', 2)
for x in range(len(te_data)):
    r = net.feedforward(te_data[x])[0]
    label = te_data_l[x]
    print(label + ": " + str(r))


#net.SGD(list(tr_data), 100, 10, 0.1, monitor_training_accuracy=True)

#net = network2.Network([9408, 500, 25, 1])
#net.SGD(list(tr_data), 100, 10, 0.5, monitor_training_accuracy=True)

#net_lu = network2.Network([12544, 600, 30, 1])
#net_lu.SGD(list(tr_data_lu), 30, 10, 0.01, monitor_training_accuracy=True)