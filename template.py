import numpy as np
import csv
import pandas as pd
for num in range(1, 2):
    read_data = np.loadtxt(rf"Competition_data\Dataset_{num}\X_train.csv", delimiter=',', skiprows=1)
    read_tar = np.loadtxt(rf"Competition_data\Dataset_{num}\Y_train.csv", delimiter=',', skiprows=1)
    lr = 0.01
    test_data = np.loadtxt(rf"Competition_data\Dataset_{num}\X_test.csv", delimiter=',', skiprows=1)
    test_tar = np.loadtxt(rf"Competition_data\Dataset_{num}\Y_predict.csv", delimiter=',', skiprows=1)
    data = read_data
    tar = read_tar

    w = np.ones(data.shape[1])
    for i in range(20000):
        y_pred = data.dot(w)

        loss = y_pred - tar
        grad = data.T.dot(loss) / data.shape[0]

        w = w - lr * grad
    y_pred = test_data.dot(w)
    modified_array = np.where(y_pred - y_pred.mean() > 0, 1, 0)
    np.savetxt(rf'Competition_data\Dataset_{num}\y_predict.csv', modified_array, delimiter=',', fmt='%d', header='y_predict', comments='')