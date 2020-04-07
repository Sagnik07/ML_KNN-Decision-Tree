#!/usr/bin/python
import numpy as np
from numpy import genfromtxt
import math

class KNNClassifier:
    def __init__(self, k=3):
	    self.k = k

    def fit_train(self, train_data, train_label):
        self.train_data = train_data
        self.train_label = train_label

    def fit_test(self, validation_data):
        self.validation_data = validation_data

    def euclidean_distance(self, row):
        dist = np.sqrt(np.sum((self.train_data-row)**2,axis=1))
        return dist

    def find_majority_label(self, k, distance):
        index = np.argsort(distance)
        index = index[:k]
        label = self.train_label[index]
        li = []
        for i in range(0,k):
            li.append(int(label[i]))

        maxm = 0
        majority_element = li[0] 
        for i in li: 
            freq = li.count(i) 
            if freq > maxm: 
                maxm = freq 
                majority_element = i

        return majority_element

    def train(self, train_path):
        data = genfromtxt(train_path, delimiter=',')
        train_data = data[:, 1:]
        train_label = data[:, 0]
        self.fit_train(train_data, train_label)

    def predict(self, test_path):
        test_data = genfromtxt(test_path, delimiter=',')
        self.fit_test(test_data)
        
        prediction = []
        
        for row in self.validation_data:
            dis = self.euclidean_distance(row) 
            prediction.append(self.find_majority_label(self.k,dis))
        
        predicted_label = np.array(prediction)
        return predicted_label
