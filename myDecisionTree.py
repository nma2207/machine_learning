#coding:utf-8

import numpy as np
import math
from sklearn import datasets as dSet
from sklearn import  metrics as sk_metrics
import matplotlib.pyplot as plt
import  time
MIN_ENTROPY=1e-2

class Node:
    #Node left
    #Node right
    #value
    #isTerminal
    #vectorOfProbs
    #k
    #index
    def __init__(self, k=0,value=None, left=None, right=None):
        self.value=value
        self.left=left
        self.right=right
        self.isTerminal=False
        self.vectorOfProbs=np.zeros((k))
        self.index=0
        self.k=k
    def setClassCount(self, k):
        self.k=k

    def fit(self, dataSet, t):
        #print 'Node fit'
        #print 'entropy =', entopy(t, self.k)
        if entopy(t, self.k)<MIN_ENTROPY:
            self.vectorOfProbs=calcClassProbs(t, self.k)
            self.isTerminal=True

        else:
            maxInfo=-1
            goodFirstSet=np.zeros((t.size))
            goodFirstT = np.zeros((t.size))
            goodSecondSet = np.zeros((t.size))

            for i in range(dataSet.shape[1]):
                points=self.__calcPoints(dataSet[:,i])
                for p in points:
                    set1,t1,set2,t2=self.__divSet(dataSet, t, i, p)
                    infoGain=self.__informationGain(t,t1,t2)
                    if(set1.size!=0 and set2.size!=0 and (maxInfo==-1 or infoGain>maxInfo)):
                        goodFirstSet=set1
                        goodFirstT=t1
                        goodSecondSet=set2
                        goodSecondT=t2
                        self.value=p
                        self.index=i
                        maxInfo=infoGain
            self.left=Node(self.k)
            self.right=Node(self.k)
            self.left.fit(goodFirstSet, goodFirstT)
            self.right.fit(goodSecondSet, goodSecondT)

    def predict(self, set):
        #print set, self.index, self.value
        if self.isTerminal==True:
            return self.vectorOfProbs

        elif set[self.index]<self.value:
            return self.left.predict(set)
        else:
            return self.right.predict(set)



    def __calcPoints(self, arr):
        np.sort(arr)
        result=[]
        for i in range(arr.size-1):
            result.append((arr[i]+arr[i+1])/2.)
        return np.array(result)

    def __divSet(self, dataSet,t, j, x):
        firstSet=[]
        firstT=[]
        secondSet=[]
        secondT=[]
        for i in range(dataSet.shape[0]):
            if dataSet[i,j]<x:
                firstSet.append(dataSet[i])
                firstT.append(t[i])
            else:
                secondSet.append(dataSet[i])
                secondT.append(t[i])
        return np.array(firstSet),np.array(firstT), \
               np.array(secondSet), np.array(secondT)

    def __informationGain(self, t, t1, t2):
        mainEntropy=entopy(t, self.k)
        entopy1=entopy(t1, self.k)
        entopy2=entopy(t2, self.k)
        return mainEntropy\
               -float(t1.size)/t.size*entopy1\
               -float(t2.size)/t.size*entopy2
    def printNode(self):
        if self.left!=None:
            self.left.printNode()
        if self.isTerminal:
            print 'I am terminal', self.vectorOfProbs
        else:
            print 'I am not terminal, index =',self.index, 'value =',self.value
        if self.right!=None:
            self.right.printNode()

class BynaryDecisionTree:
    #Node head
    #int k- count of class
    def __init__(self):
        self.__head=None
        self.k=0
    def addValue(self, value):
        if self.__head is None:
            self.__head=Node(value)
        else:
            p=self.__head
            prev=p
            while p!=None:
                prev=p
                if(p.value>value):
                    p=p.left
                else:
                    p=p.right
            new=Node(value)
            if prev.value>value:
                prev.left=new
            else:
                prev.right=new
    def printTree(self):
        if self.__head !=None:
            self.__head.printNode()
        else:
            print 'EMpty tree'
    def fit(self, dataSet, t):
        #self.__head.setClassCount(t.size)
        self.__head=Node(np.max(t)+1)
        self.__head.fit(dataSet, t)

    def predict(self, data):
        result=[]
        for i in range(data.shape[0]):
            result.append(self.__head.predict(data[i]))
        return np.array(result)
        # else:
        #     return np.array(self.__head.predict(data))

def entopy(t, k):
    count=np.zeros((k))
    for i in t:
        count[i]+=1
    indexs=np.where(count!=0)[0]
    count=count[indexs]
    N=t.size
    #print count, N, (np.log2(np.float64(count)))
    result=-np.sum(count*np.log2(np.float64(count/N)))/N
    return result

def calcClassProbs(t, k):
    result=np.zeros((k))
    for i in t:
        result[i]+=1
    result/=float(np.sum(result))
    return result

def calcClassByProbs(t):
    result=np.zeros((t.shape[0]))
    for i in range(t.shape[0]):
        k=np.where(t[i]==np.max(t[i]))[0][0]
        result[i]=int(k)
    return result

def calc_accuracy(tree, data, real_t, k):
    predict_t=calcClassByProbs(tree.predict(data))
    accuracy_matrix=np.zeros((k,k))
    #print predict_t
    for i in range(data.shape[0]):
        accuracy_matrix[int(predict_t[i]), int(real_t[i])]+=1
    print accuracy_matrix
    result=0
    for i in range(k):
        result+=accuracy_matrix[i,i]
    return result/float(np.sum(accuracy_matrix))

def main():
    tree=BynaryDecisionTree()
    #Пример с лекции Евгения Викторовича с собакой, кошкой и уткой
    # data=np.array([[0,4,1],
    #                [0,4,0],
    #                [0,2,0],
    #                [1,2,0]])
    # t=np.array([0,1,2,3])
    # tree.fit(data,t)
    # print 'end'
    # #tree.printTree()
    # print calc_accuracy(tree, data, t, 4)
    digits=dSet.load_digits()
    imgs=digits.images[:1000]
    target=digits.target[:1000]
    data=digits.data[:1000]
    start=time.time()
    tree.fit(data, target)
    end=time.time()
    print start,end, end-start
    print 'end'
    exper=digits.data[1000:]
    exp_t=digits.target[1000:]
    t=tree.predict(exper)

    print calc_accuracy(tree, exper, exp_t, 10)
    t=calcClassByProbs(t)
    print sk_metrics.accuracy_score(exp_t, t)

if __name__=="__main__":
    main()