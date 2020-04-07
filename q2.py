#!/usr/bin/python
import numpy as np
from numpy import genfromtxt
import math
import pandas as pd 
from pandas import DataFrame
from collections import Counter

class KNNClassifier:
    def __init__(self, k=5):
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
        majority_label = Counter(label).most_common(1)
        return majority_label[0][0]

    def one_hot_encoding_validation(self, validation_data1):
        attributes = [['b','c','x','f','k','s'],['f','g','y','s'],['n','b','c','g','r','p','u','e','w','y'],['t','f'],
              ['a','l','c','y','f','m','n','p','s'],['a','d','f','n'],['c','w','d'],['b','n'],
              ['k','n','b','h','g','r','o','p','u','e','w','y'],['e','t'],['b','c','u','e','z','r'],['f','y','k','s'],['f','y','k','s'],
              ['n','b','c','g','o','p','e','w','y'],['n','b','c','g','o','p','e','w','y'],['p','u'],
              ['n','o','w','y'],['n','o','t'],['c','e','f','l','n','p','s','z'],['k','n','b','h','r','o','u','w','y'],
              ['a','c','n','s','v','y'],['g','l','m','p','u','w','d']]
        j=0
        validation_df = pd.DataFrame()
        # validation_data.shape
        for i in validation_data1.T:
            dummies = pd.get_dummies(i, prefix='', prefix_sep='')
            dummies = dummies.T.reindex(attributes[j]).T.fillna(0)
            j = j+1
            validation_df = pd.concat([validation_df,dummies],axis=1, sort=False)
        validation_df1 = validation_df.to_numpy()
        return validation_df1

    def one_hot_encoding_train(self, train_data1):
        attributes = [['b','c','x','f','k','s'],['f','g','y','s'],['n','b','c','g','r','p','u','e','w','y'],['t','f'],
              ['a','l','c','y','f','m','n','p','s'],['a','d','f','n'],['c','w','d'],['b','n'],
              ['k','n','b','h','g','r','o','p','u','e','w','y'],['e','t'],['b','c','u','e','z','r'],['f','y','k','s'],['f','y','k','s'],
              ['n','b','c','g','o','p','e','w','y'],['n','b','c','g','o','p','e','w','y'],['p','u'],
              ['n','o','w','y'],['n','o','t'],['c','e','f','l','n','p','s','z'],['k','n','b','h','r','o','u','w','y'],
              ['a','c','n','s','v','y'],['g','l','m','p','u','w','d']]
        j=0
        train_df = pd.DataFrame()
        for i in train_data1.T:
            dummies = pd.get_dummies(i, prefix='', prefix_sep='')
            dummies = dummies.T.reindex(attributes[j]).T.fillna(0)
            j = j+1
            train_df=pd.concat([train_df,dummies],axis=1, sort=False)
        train_df1 = train_df.to_numpy()
        return train_df1

    def train(self, train_path):
        train_df=pd.read_csv(train_path,header=None)
        for col in train_df.columns:
            train_df[col]=train_df[col].replace('?',train_df[col].mode()[0])
        data = train_df.to_numpy()
        train_data1 = data[:, 1:]
        train_label = data[:, 0]
        train_data = self.one_hot_encoding_train(train_data1)
        self.fit_train(train_data, train_label)

    def predict(self, test_path):
        test_df=pd.read_csv(test_path,header=None)
        for col in test_df.columns:
            test_df[col]=test_df[col].replace('?',test_df[col].mode()[0])
        validation_data1 = test_df.to_numpy()
        validation_data = self.one_hot_encoding_validation(validation_data1)
        self.fit_test(validation_data)
        
        prediction = []
        
        for row in self.validation_data:
            dis = self.euclidean_distance(row) 
            prediction.append(self.find_majority_label(self.k,dis))
        
        predicted_label = np.array(prediction)
        return predicted_label
