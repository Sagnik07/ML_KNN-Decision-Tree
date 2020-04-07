#!/usr/bin/python
import numpy as np
from numpy import genfromtxt
import math
import pandas as pd
from pandas import DataFrame
from collections import Counter
import sys

class Node:
    def __init__(self, attribute, split, ans):
        self.attr = attribute
        self.split_point = split
        self.answer = ans
        self.left = None
        self.right = None

class DecisionTree:
    def __init__(self):
	    pass

    def fit_train(self, data, global_avg):
        self.data = data
        self.global_avg = global_avg

    def fit_test(self, test_data):
        self.test_data = test_data

    def rms(self, dataset, element):
        mean_sp = dataset['SalePrice'].mean()
        temp = (dataset['SalePrice'] - mean_sp)**2
        sum1 = temp.values.sum()
	    # sum1 = sum1/(len(self.data.index))
        return math.sqrt(sum1/(len(self.data.index)))

    def split_n_rms(self, left_dataset, right_dataset, element):
        left_rms = self.rms(left_dataset, element)
        right_rms = self.rms(right_dataset, element)
        left_rms_length = len(left_dataset.index)
        right_rms_length = len(right_dataset.index)
        total_rms_length = left_rms_length + right_rms_length
	#left_rms = math.sqrt((left_rms/left_rms_length))
	#right_rms = math.sqrt((right_rms/right_rms_length))
        rms_value = ((left_rms_length/total_rms_length)*left_rms) + ((right_rms_length/total_rms_length)*right_rms)
        return rms_value

    def find_node(self, data):
        min_rms = sys.maxsize
        split_attr = ""
        split_point = 0
        for col in data.columns:
            if(col == 'SalePrice'):
                break
            li = data[col].tolist()
            li1 = list(set(li))
            if(isinstance(li1[0],str)):
                for element in li1:
                    left_dataset = data[data[col] == element]
                    right_dataset = data[data[col] != element]
                    rms_value = self.split_n_rms(left_dataset, right_dataset, element)
                    if(rms_value<=min_rms):
                        split_attr = col
                        split_point = element
                        min_rms = rms_value
            else:
                for i in range(0,len(li1)-1):
                    element = (li1[i] + li1[i+1])/2
                    left_dataset = data[data[col]<=element]
                    right_dataset = data[data[col]>element]
                    rms_value = self.split_n_rms(left_dataset, right_dataset, element)
                    if(rms_value<=min_rms):
                        split_attr = col
                        split_point = element
                        min_rms = rms_value
                
        return split_attr, split_point

    def makeTree(self, data, level):
        if(level >= 10):
            avg = data['SalePrice'].mean()
            root = Node('SalePrice',-1,avg)
            return root
        if(len(data.index)==0):
            root = Node('SalePrice',-1,self.global_avg)
            return root
        if(len(data.index)<=50):
            avg = data['SalePrice'].mean()
            root = Node('SalePrice',-1,avg)
            return root
        node_attr, node_split = self.find_node(data)
        root = Node(node_attr, node_split, 0)
        left_dataframe = data[data[node_attr]<=node_split]
        left_dataframe = left_dataframe.drop([node_attr],axis=1)
        right_dataframe = data[data[node_attr]>node_split]
        right_dataframe = right_dataframe.drop([node_attr],axis=1)
        root.left = self.makeTree(left_dataframe, level+1)
        root.right = self.makeTree(right_dataframe, level+1)
        return root

    def train(self, train_path):
        data = pd.read_csv(train_path)
        data = data.drop(['Id'], axis=1)
        data.dropna(thresh=data.shape[0]*0.8, how='all', axis=1, inplace=True)

        for col in data.columns:
            li = data[col].tolist()
            if(isinstance(li[0], str)):
                data[col].fillna(data[col].mode()[0], inplace=True)
            else:
                data[col].fillna(data[col].mean(), inplace=True)

        data1 = data
        global_avg = data['SalePrice'].mean()
        self.fit_train(data, global_avg)

    def search(self, root, row):
        if(root == None):
            return
        if(root.left == None and root.right==None):
            return root.answer
        
        attribute = root.attr
        li = self.data[attribute].tolist()
        if(isinstance(li[0],str)):
            value = row[attribute]
            if(value == root.split_point):
                return self.search(root.left,row)
            else:
                return self.search(root.right,row)
        else:
            value = row[attribute]
            if(value <= root.split_point):
                return self.search(root.left,row)
            else:
                return self.search(root.right,row)

    def predict(self, test_path):
        test_data = pd.read_csv(test_path)
        self.fit_test(test_data)

        min_rms = sys.maxsize
        root = self.makeTree(self.data, 0)
        li = []
        for i in range(0,len(self.test_data)):
            row = test_data.iloc[i]
            ans = self.search(root,row)
            li.append(ans)
            # print("ans: ",ans)
        predicted_label = np.asarray(li)
        return predicted_label










