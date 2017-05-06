import numpy as np
import math

class Node:
    #Node left
    #Node right
    #value
    #isTerminal
    #vectorOfProbs
    def __init__(self, value=None,k=0, left=None, right=None):
        self.value=value
        self.left=left
        self.right=right
        self.isTerminal=False
        self.vectorOfProbs=np.zeros((k))


    def printNode(self):
        if self.left!=None:
            self.left.printNode()
        print self.value
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
        #TODO
        return 0
    def predict(self, data):
        #TODO
        return 0




def main():
    t=BynaryDecisionTree()
    t.addValue(5)
    t.printTree()
    t.addValue(4)
    t.addValue(1)
    t.addValue(7)
    t.addValue(19)
    t.addValue(11)

    t.printTree()

if __name__=="__main__":
    main()