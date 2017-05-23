# -*- coding: utf-8 -*-

from __future__ import print_function
import sklearn.datasets
import matplotlib.pyplot as plt
import numpy as np
import random
import numpy.linalg as alg


class k_means_cluster:
    # int k
    # list of centers
    # exit_criterion defoult - max_dist//n
    def __init__(self, k=0, exit_criterion='max_dist'):
        self.k = k
        self.list_of_centers = []
        self.exit_criterion=exit_criterion

    def __initCenters(self, data):
        a = np.arange(data.shape[0])
        np.random.shuffle(a)
        for i in range(self.k):
            self.list_of_centers.append(data[a[i]])

    def fit(self, data, criterion):
        self.__initCenters(data)
        i=0
        while(True):
            #print(float(i) / N * 100, '%')
            new_center = []
            k_count = []
            for j in range(self.k):
                new_center.append(np.zeros(2))
                k_count.append(0)
            for d in data:
                # print(d.shape)
                # print(d)
                dist = []
                for cent in self.list_of_centers:
                    # print(cent)
                    dist.append(alg.norm(cent-d))
                dist = np.array(dist)
                k_num = np.where(dist == np.min(dist))
                # print(dist)
                # print(k_num)
                k_num = k_num[0][0]
                new_center[k_num] += d
                k_count[k_num] += 1
            for j in range(self.k):
                if (k_count[j] != 0):
                    new_center[j] /= k_count[j]

            if self.exit_criterion=='n':
                if i>=criterion:
                    self.list_of_centers = new_center
                    break
                else:
                    print(float(i) / criterion * 100, '%')
            elif self.exit_criterion=='max_dist':
                old=np.array(self.list_of_centers)
                new=np.array(new_center)
                dif=old-new
                dist=np.sqrt(dif[:,0]**2+dif[:,1]**2)
                max_dist=alg.norm(dist, np.inf)
                if max_dist<criterion:
                    self.list_of_centers = new_center
                    break
                else:
                    print('max_dist =',max_dist)

            self.list_of_centers = new_center
            i+=1
            # print(self.list_of_centers)

    def predict(self, data):
        result = []
        for d in data:
            dist = []
            for center in self.list_of_centers:
                dist.append(alg.norm(d-center))
            dist = np.array(dist)
            k = np.where(dist == np.min(dist))[0][0]
            result.append(k)
        return np.array(result)


def main():
    x, y = sklearn.datasets.make_blobs(n_samples=1000, cluster_std=2, centers=3)
    #print(x.shape, y)
    k=3
    k_m = k_means_cluster(k)
    k_m.exit_criterion='max_dist'
    k_m.fit(x, 1e-6)
    result = k_m.predict(x)
    dots=[]
    for _ in range(k):
        l=[]
        dots.append(l)
    for i in range(result.size):
        dots[result[i]].append(x[i])
    plt.figure()
    for dot in dots:
        d=np.array(dot)
        plt.plot(d[:,0], d[:,1],'.', color=(random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)))
    for center in k_m.list_of_centers:
        plt.plot(center[0],center[1],'xk')
    plt.show()


if __name__ == "__main__":
    main()