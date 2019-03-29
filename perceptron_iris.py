import numpy as np
import random
import math

TRAINING_PERCENTAGE = 0.05
TEST_PERCENTAGE = 1 - TRAINING_PERCENTAGE

class Perceptron(object):
    def __init__(self, input_size, lr=1, epochs=100):
        self.W = np.zeros(input_size+1)
        # add one for bias
        self.epochs = epochs
        self.lr = lr
    
    def activation_fn(self, x):
        #return (x >= 0).astype(np.float32)
        #print( " x=", x)
        return 1 if x >= 0 else 0
 
    def predict(self, x):
        z = self.W.T.dot(x)
        a = self.activation_fn(z)
        return a
 
    def fit(self, X, d):
        for _ in range(self.epochs):
            for i in range(d.shape[0]):
                x = np.insert(X[i], 0, 1)
                y = self.predict(x)
                e = d[i] - y
                self.W = self.W + self.lr * e * x

                

if __name__ == '__main__':
    iris_labels = ["Iris-setosa", "Iris-virginica", "Iris-Versicolor"]
    random_index = random.randint(0,2)
    del iris_labels[random_index]

    label_to_int = {}
    int_to_label = {}

    for index, label in enumerate(iris_labels):
        label_to_int[label] = index
        int_to_label[index] = label

    iris_data = open("iris.data", "r")

    lines = iris_data.readlines()

    array_data = []
    decision = []

    for line in lines:
        data = line.split(",")
        label = data[-1].replace("\n", "")
        if label in label_to_int:
            del data[-1]
            array_data.append(list(map(float, data)))
            decision.append(label_to_int[label])
    
    data_size = len(array_data)
    training_data_size = math.ceil(data_size * TRAINING_PERCENTAGE)
    test_data_size = math.floor(data_size * TEST_PERCENTAGE)

    print(data_size)
    print(training_data_size)
    print(test_data_size)

    training_sample_indexes = set(random.sample(range(0, data_size), training_data_size))
    test_sample_indexes = set(range(0, data_size)) - training_sample_indexes

    training_data = []
    training_data_result = []

    for index in training_sample_indexes:
        training_data.append(array_data[index])
        training_data_result.append(decision[index])
    
    test_data = []
    test_data_result = []
    
    for index in test_sample_indexes:
        test_data.append(array_data[index])
        test_data_result.append(decision[index])

    X = np.array(training_data)
    d = np.array(training_data_result)
 
    perceptron = Perceptron(input_size=len(training_data[0]))
    perceptron.fit(X, d)
    print("The W results = ",perceptron.W)

    right_predictions_count = 0


    for index, test in enumerate(test_data):
        x = np.insert(test, 0, 1)
        prediction = perceptron.predict(x)

        print("------")
        print("The input value (x) =", test)
        print("The predict - y = ", int_to_label[prediction])
        print("The real value = ", int_to_label[test_data_result[index]] )
        print("------")

        if (prediction == test_data_result[index]):
            right_predictions_count += 1
    
    print("\n\n\n*******")
    print("Number of tests = ", len(test_data))
    print("Number of right predictions = ", right_predictions_count)
    print("Number of wrong predictions = ", len(test_data) - right_predictions_count)
    print("Correct prediction rate = ", (right_predictions_count/len(test_data) * 100), "%")
    print("*******\n\n\n")
