import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x, grad_y):
    return grad_y * sigmoid(x) * (1 - sigmoid(x))

def matrix_right_mul_W_derivative(X, grad_Y):
    return  X.T @ grad_Y    

def matrix_right_mul_X_derivative(W, grad_Y):
    return  grad_Y @ W.T  

def matrix_plus_derivative(grad_Y, batch_size):
    return np.sum(grad_Y, axis=0) / batch_size


class simpleNN:
    def __init__(self, batch_size, hiddenLayerSize, feature_size):
        self.lr = 0.08
        self.batch_size = batch_size
        self.feature_size = feature_size
        self.W1 = np.random.uniform(0, 1, (feature_size, hiddenLayerSize[0]))
        self.W2 = np.random.uniform(0, 1, (hiddenLayerSize[0], hiddenLayerSize[1]))
        self.W3 = np.random.uniform(0, 1, (hiddenLayerSize[1], 1))

        self.b1 = np.zeros(hiddenLayerSize[0])
        self.b2 = np.zeros(hiddenLayerSize[1])
        self.b3 = np.zeros(1)

        self.Z1 = np.zeros((batch_size, hiddenLayerSize[0]))
        self.Z2 = np.zeros((batch_size, hiddenLayerSize[1]))
        self.Z3 = np.zeros((batch_size, 1))
        
        self.a1 = np.zeros((batch_size, hiddenLayerSize[0]))
        self.a2 = np.zeros((batch_size, hiddenLayerSize[1]))
        self.a3 = np.zeros((batch_size, 1))
    

    def forward(self, X, batch_size = None):
        
        if batch_size == None:
            batch_size = self.batch_size
        # Layer 1
        self.Z1 = (X@self.W1 + np.tile(self.b1, (batch_size, 1)))/ X.shape[1]
        self.a1 = sigmoid(self.Z1)
        # Layer 2
        self.Z2 = (self.a1@self.W2 + np.tile(self.b2, (batch_size, 1))) / self.a1.shape[1]
        self.a2 = sigmoid(self.Z2)
        # Layer 3
        self.Z3 = (self.a2@self.W3 + np.tile(self.b3, (batch_size, 1))) / self.a2.shape[1]
        self.a3 = sigmoid(self.Z3)  

        return self.a3
    

    def criterion(self, tar, pred_y):
        loss =  - (tar.T @ np.log(pred_y) + (1 - tar).T @ np.log(1 - pred_y)) / self.batch_size 
        return loss[0][0]

    def backproapgation(self, X, tar, pred_y):
        
        grad_a3_pred = - (tar / pred_y - (1 - tar) / (1 - pred_y))  # Loss: - (y_hat * np.log(pred_y) + (1 - y_hat) * np.log(1 - pred_y))
        grad_Z3_a3 = sigmoid_derivative(self.Z3, grad_a3_pred)          # sigmoid(Z3)
        grad_a2_Z3_W3 = matrix_right_mul_W_derivative(self.a2, grad_Z3_a3)  # XW + b, cacualte W
        grad_a2_Z3_b3 = matrix_plus_derivative(grad_Z3_a3, self.batch_size) # XW + b, cacualte b
        grad_a2_Z3 = matrix_right_mul_X_derivative(self.W3, grad_Z3_a3)     # XW + b, cacualte X

        self.W3 -= grad_a2_Z3_W3 * self.lr
        self.b3 -= grad_a2_Z3_b3 * self.lr

        grad_Z2_a2 = sigmoid_derivative(self.Z2, grad_a2_Z3)            # sigmoid(Z2)
        grad_a1_Z2_W2 = matrix_right_mul_W_derivative(self.a1, grad_Z2_a2)  # a1*W2 + b2, cacualte W 
        grad_a1_Z2_b2 = matrix_plus_derivative(grad_Z2_a2, self.batch_size) # a1*W2 + b2, cacualte b
        grad_a1_Z2 = matrix_right_mul_X_derivative(self.W2, grad_Z2_a2)     # a1*W2 + b2, cacualte X

        self.W2 -= grad_a1_Z2_W2 * self.lr
        self.b2 -= grad_a1_Z2_b2 * self.lr

        grad_Z1_a1 = sigmoid_derivative(self.Z1, grad_a1_Z2)            # sigmoid(Z1)
        grad_X_Z1_W1 = matrix_right_mul_W_derivative(X, grad_Z1_a1)        # X*W1 + b1, cacualte W 
        grad_X_Z1_b1 = matrix_plus_derivative(grad_Z1_a1, self.batch_size) # X*W1 + b1, cacualte b

        self.W1 -= grad_X_Z1_W1 * self.lr
        self.b1 -= grad_X_Z1_b1 * self.lr
    
    def test(self, X_test, sample_size, num):
        prediction = self.forward(X_test, sample_size)
        # prediction_cls = (prediction > 0.5).astype(int)
        # eval = np.equal(prediction_cls, Y_test)
        # accuracy = np.sum(eval) / sample_size
        np.savetxt(rf'Competition_data\Dataset_{num}\y_predict.csv', prediction, delimiter=',', fmt='%d', header='y_predict', comments='')

    def train(self, data, target, n_epoch, sample_size, batch_size, X_test, Y_test, sample_size_test):
        loss_check_point_x = []
        loss_check_point_y = []
        for epoch in range(n_epoch):
            for batch in range( np.ceil(sample_size / batch_size).astype(int) ):
                inputs = data[batch*batch_size : min(sample_size, (batch+1)*batch_size), :]
                labels = target[batch*batch_size : (batch+1)*batch_size, :]
                prediction = self.forward(inputs)
                loss = self.criterion(labels, prediction)
                self.backproapgation(inputs, labels, prediction)
            if (epoch+1) % 500 == 0 or epoch==0:
                print(epoch, end = " ")
                # self.test(X_test, Y_test, sample_size_test)

for num in range(1, 50):
    if(num in [3, 5, 7, 11, 12, 22, 25, 36, 41, 45, 47]):
        read_data = np.loadtxt(rf"Competition_data\Dataset_{num}\X_train.csv", delimiter=',', skiprows=1)
        data_test = np.loadtxt(rf"Competition_data\Dataset_{num}\X_test.csv", delimiter=',', skiprows=1)
        read_tar = np.loadtxt(rf"Competition_data\Dataset_{num}\Y_train.csv", delimiter=',', skiprows=1)
        data = read_data.astype(float)
        tar = read_tar
        tar = tar.reshape(np.shape(tar)[0], 1)

        min_values = np.min(data, axis=0)
        max_values = np.max(data, axis=0)
        cols_to_keep = max_values != min_values
        data = data[:, cols_to_keep]
        scaled_data = (data - min_values[cols_to_keep]) / (max_values[cols_to_keep] - min_values[cols_to_keep])
        data = np.clip(scaled_data, 0.0001, 0.9999)

        min_values_test = np.min(data_test, axis=0)
        max_values_test = np.max(data_test, axis=0)
        data_test = data_test[:, cols_to_keep]
        scaled_data_test = (data_test - min_values[cols_to_keep]) / (max_values[cols_to_keep] - min_values[cols_to_keep])
        data_test = np.clip(scaled_data_test, 0.0001, 0.9999)

        sample_size = np.shape(data)[0] - np.shape(data)[0] % 4
        batch_size = 4
        n_epoch = 200
        model = simpleNN(batch_size, [10,10], np.shape(data)[1])
        model.train(data[:sample_size, :], tar[:sample_size, :], n_epoch, sample_size, batch_size, data[120:sample_size, :], tar[120:sample_size], sample_size )
        model.test(data_test, np.shape(data_test)[0], num)
        print(num)