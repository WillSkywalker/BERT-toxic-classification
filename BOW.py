import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.feature_extraction.text import CountVectorizer

#Opening files
file = open("/users/Jacqueline/Documents/Language_Technology_Project/train_preprocessed.csv")
content = file.readlines()
file.close()
data = [line.split(",") for line in content] 
for x in data:
    del x[1]
    del x[4]

comments = []
for x in data[1:]:
    y = x[0]
    comments.append(y)

#vectorize
vectorizer = CountVectorizer()
vectors = vectorizer.fit_transform(comments)    
    
#split in train and test
train= []
test = []
for x in vectors[:143615]:
    train.append(y)

for x in vectors[143615:]:
    test.append(y)

#Neural Network

def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1],4) 
        self.weights2   = np.random.rand(4,1)                 
        self.y          = y
        self.output     = np.zeros(y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))
        
    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2

if __name__ == "__main__":
    X = vectors[:143615]
    y = vectors[143615:]
    nn = NeuralNetwork(X,y)

    for i in range(1500):
        nn.feedforward()
        nn.backprop()

    print(nn.output)
