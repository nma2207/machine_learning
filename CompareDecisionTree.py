#coding:utf-8
import  numpy as np
import sklearn.tree
from sklearn import datasets as dSet
import time
from sklearn import  metrics as sk_metrics
import sklearn.ensemble

#
#ВНИМАНИЕ!!! ИСПОЛЬЗУЕТСЯ Python 2.7
#
def main():
    # tree=sklearn.tree.DecisionTreeClassifier(criterion='entropy')
    # tree.min_impurity_split=1e-2
    # tree.max_features=None
    digits=dSet.load_digits()
    imgs=digits.images
    n=imgs.shape[0] #=1797
    # target=digits.target[:1000]
    # data=digits.data[:1000]
    #best_tree=sklearn.tree.DecisionTreeClassifier()
    train_data=digits.data[:1200]
    train_t=digits.target[:1200]

    valid_data=digits.data[1200:1500]
    valid_t=digits.target[1200:1500]

    test_data=digits.data[1500:]
    test_t=digits.target[1500:]
    max_accuracy=0
    best_params={}
    best_forest=sklearn.ensemble.RandomForestClassifier()
    for tree_count in range(1, 20):
        for min_impurt_split in np.linspace(1e-09, 1e-1, 10):
            for height in np.arange(10, 20, 1):
                for criterion in ['entropy','gini']:
                    print tree_count,min_impurt_split, height, criterion
                    #tree=sklearn.tree.DecisionTreeClassifier(criterion=criterion)
                    forest=sklearn.ensemble.RandomForestClassifier(tree_count)
                    forest.min_impurity_split=min_impurt_split
                    forest.max_depth=height
                    forest.fit(train_data, train_t)
                    exp_t=forest.predict(valid_data)
                    accuracy=sk_metrics.accuracy_score(valid_t, exp_t)
                    if(accuracy>max_accuracy):
                        max_accuracy=accuracy
                        #best_tree=tree
                        best_forest=forest
                        best_params={
                            'tree_count':tree_count,
                            'min_impurity_split':min_impurt_split,
                            'max_depth':height,
                            'criterion':criterion
                        }

    exp_test_t=best_forest.predict(test_data)
    test_accuracy=sk_metrics.accuracy_score(exp_test_t, test_t)
    print test_accuracy
    print best_params


if __name__=="__main__":
    main()