#coding:utf-8
import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg as nalg
#русские комментарии

#n - степень полинома
def computing_F(x,n):
    F=np.ones((x.size, n+1))
    for i in range(1,n+1):
        F[:,i]=x**i
    return F


def computing_w(F,t,lam=0):
    I=np.identity(F.shape[1])
    return nalg.inv((F.transpose().dot(F))+lam*I).dot(F.transpose()).dot(t)





def main():
    x = np.linspace(0, 1, 1000)
    y = 20 * np.sin(2 * np.pi * 3 * x) + 100 * np.exp(x)
    e = 10 * np.random.randn(1000)
    t = y + e
    k = 1
    e_count=10
    errors=np.zeros((e_count-1))
    for e in range(1, e_count):
        F = computing_F(x, e)
        w=computing_w(F,t,13)
        new_y = F.dot(w)
        errors[e-1] = np.sum((new_y - t) ** 2)
        # print(np.where(errors == np.min(errors)))
        print errors[e-1]
    i=np.where(errors==np.min(errors))
    print np.min(errors), i
    index=i[0][0]
    F = computing_F(x, index)
    w = computing_w(F, t, 0)
    new_y = F.dot(w)
    plt.figure()
    #plt.plot(np.arange(k), errors, 'g')
    plt.plot(x,t, '.g')
    plt.plot(x,y,'r')
    plt.plot(x,new_y,'b')
    plt.show()

if __name__=="__main__":
    main()