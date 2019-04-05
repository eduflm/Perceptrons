import numpy as np
import sys

class Perceptron(object):
    def __init__(self, input_size, lr=1, epochs=100):
        self.W = np.zeros(input_size+1)
        self.epochs = epochs
        self.lr = lr
    
    def activation_fn(self, x):
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
    
    training_array = [[0,0], [0,1], [1,0], [1,1]]
    not_training_array = [[0],[1]]
    
    and_answers = [0,0,0,1]
    or_answers = [0,1,1,1]
    not_answers= [1,0]
    xor_expected_answers = [0,1,1,0]


    #Training AND perceptron

    training_array = np.array(training_array)
    and_answers = np.array(and_answers)

    and_perceptron = Perceptron(input_size=len(training_array[0]))
    and_perceptron.fit(training_array, and_answers)
    print("The W results for AND perceptron = ",and_perceptron.W)

    #Training OR perceptron

    training_array = np.array(training_array)
    or_answers = np.array(or_answers)

    or_perceptron = Perceptron(input_size=len(training_array[0]))
    or_perceptron.fit(training_array, or_answers)
    print("The W results for OR perceptron = ",or_perceptron.W)

    #Training NOT perceptron

    not_training_array = np.array(not_training_array)
    not_answers = np.array(not_answers)

    not_perceptron = Perceptron(input_size=len(not_training_array[0]))
    not_perceptron.fit(not_training_array, not_answers)
    print("The W results for NOT perceptron = ",not_perceptron.W)

    # Start XOR
    for index,entry in enumerate(training_array):

        or_entry = np.insert(entry, 0 , 1)
        or_prediction = or_perceptron.predict(or_entry)

        and_entry = np.insert(entry, 0 , 1)
        and_prediction = and_perceptron.predict(and_entry)

        not_and_entry = np.insert([and_prediction], 0 , 1)
        not_and_prediction = not_perceptron.predict(not_and_entry)

        xor_entry = np.insert([or_prediction, not_and_prediction], 0 , 1)
        xor_prediction = and_perceptron.predict(xor_entry)

        print("\n\n*****")
        print("The entry was: ", entry)
        print("The prediction was: ", xor_prediction)
        print("The correct answer is: ", xor_expected_answers[index])
        print("*****\n\n")
