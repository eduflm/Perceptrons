import numpy as np

class Perceptron(object):
    def __init__(self, input_size, lr=1, epochs=100, isBipolarActivation = False):
        self.W = np.zeros(input_size+1)
        self.epochs = epochs
        self.lr = lr
        self.isBipolarActivation = isBipolarActivation
    
    def activation_fn(self, x):
        if self.isBipolarActivation:
            return 1 if x >= 0 else -1
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
    
    binary_training_array = np.array([[0,0], [0,1], [1,0], [1,1]])
    binary_and_answers = np.array([0,0,0,1])
    binary_or_answers = np.array([0,1,1,1])


    bipolar_training_array = np.array([[-1,-1], [-1,1], [1,-1], [1,1]])
    bipolar_and_answers = np.array([-1,-1,-1,1])
    bipolar_or_answers = np.array([-1,1,1,1])


    #Training AND binary perceptron
    binary_and_perceptron = Perceptron(input_size=len(binary_training_array[0]))
    binary_and_perceptron.fit(binary_training_array, binary_and_answers)
    print("The W results for binary AND perceptron = ",binary_and_perceptron.W)

    #Training AND bipolar PERCEPTRON
    bipolar_and_perceptron = Perceptron(input_size=len(bipolar_training_array[0]), isBipolarActivation = True)
    bipolar_and_perceptron.fit(bipolar_training_array, bipolar_and_answers)
    print("The W results for bipolar AND perceptron = ",bipolar_and_perceptron.W)

    #Trainong AND with binary input and bipolar output
    binary_bipolar_and_perceptron = Perceptron(input_size=len(binary_training_array[0]), isBipolarActivation=True)
    binary_bipolar_and_perceptron.fit(binary_training_array, bipolar_and_answers)
    print("The W results for AND with binary input and bipolar output perceptron = ",binary_bipolar_and_perceptron.W)


    



      #Training OR binary perceptron
    binary_or_perceptron = Perceptron(input_size=len(binary_training_array[0]))
    binary_or_perceptron.fit(binary_training_array, binary_or_answers)
    print("The W results for binary or perceptron = ",binary_or_perceptron.W)

    #Training OR bipolar perceptron
    bipolar_or_perceptron = Perceptron(input_size=len(bipolar_training_array[0]), isBipolarActivation=True)
    bipolar_or_perceptron.fit(bipolar_training_array, bipolar_or_answers)
    print("The W results for bipolar or perceptron = ",bipolar_or_perceptron.W)

    #Trainong OR with binary input and bipolar output
    binary_bipolar_or_perceptron = Perceptron(input_size=len(binary_training_array[0]), isBipolarActivation=True)
    binary_bipolar_or_perceptron.fit(binary_training_array, bipolar_or_answers)
    print("The W results for OR with binary input and bipolar output perceptron = ",binary_bipolar_or_perceptron.W)


    #Testing binary results
    for index in range(0,4):

        binary_and_entry = np.insert(binary_training_array[index], 0 , 1)
        binary_and_prediction = binary_and_perceptron.predict(binary_and_entry)
        print("\n\n*****")
        print("Entrada binária e saída binária para o AND")
        print("The entry was: ", binary_training_array[index])
        print("The prediction was: ", binary_and_prediction)
        print("The correct answer is: ", binary_and_answers[index])
        print("*****\n\n")


        bipolar_and_entry = np.insert(bipolar_training_array[index], 0 , 1)
        bipolar_and_prediction = bipolar_and_perceptron.predict(bipolar_and_entry)
        print("\n\n*****")
        print("Entrada bipolar e saída bipolar para o AND")
        print("The entry was: ", bipolar_training_array[index])
        print("The prediction was: ", bipolar_and_prediction)
        print("The correct answer is: ", bipolar_and_answers[index])
        print("*****\n\n")

        bipolar_binary_and_entry = np.insert(binary_training_array[index], 0 , 1)
        bipolar_binary_and_prediction = binary_bipolar_and_perceptron.predict(bipolar_binary_and_entry)
        print("\n\n*****")
        print("Entrada binária e saídas bipolar para o AND")
        print("The entry was: ", binary_training_array[index])
        print("The prediction was: ", bipolar_binary_and_prediction)
        print("The correct answer is: ", bipolar_and_answers[index])
        print("*****\n\n")




        binary_or_entry = np.insert(binary_training_array[index], 0 , 1)
        binary_or_prediction = binary_or_perceptron.predict(binary_or_entry)
        print("\n\n*****")
        print("Entrada binária e saída binária para o OR")
        print("The entry was: ", binary_training_array[index])
        print("The prediction was: ", binary_or_prediction)
        print("The correct answer is: ", binary_or_answers[index])
        print("*****\n\n")


        bipolar_or_entry = np.insert(bipolar_training_array[index], 0 , 1)
        bipolar_or_prediction = bipolar_or_perceptron.predict(bipolar_or_entry)
        print("\n\n*****")
        print("Entrada bipolar e saída bipolar para o OR")
        print("The entry was: ", bipolar_training_array[index])
        print("The prediction was: ", bipolar_or_prediction)
        print("The correct answer is: ", bipolar_or_answers[index])
        print("*****\n\n")

        bipolar_binary_or_entry = np.insert(binary_training_array[index], 0 , 1)
        bipolar_binary_or_prediction = binary_bipolar_or_perceptron.predict(bipolar_binary_or_entry)
        print("\n\n*****")
        print("Entrada binária e saídas bipolar para o OR")
        print("The entry was: ", binary_training_array[index])
        print("The prediction was: ", bipolar_binary_or_prediction)
        print("The correct answer is: ", bipolar_or_answers[index])
        print("*****\n\n")

