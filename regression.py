#coding:utf-8
import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg as nalg


#  ВНИМАНИЕ!!!!
#  ИСПОЛЬЗУЕТСЯ Python 2.7

#n - степень полинома
def computing_F(x,n):
    F=np.ones((x.size, n+1))
    for i in range(1,n+1):
        F[:,i]=x**i
    return F


def computing_w(F,t,lam=0):
    I=np.identity(F.shape[1])
    return nalg.inv((F.transpose().dot(F))+lam*I).dot(F.transpose()).dot(t)

def computing_y(w,x):
    y=np.zeros((x.size), dtype=float)
    for i in range(w.size):
        y+=w[i]*(x**i)
    return y

def find_min_error(x,t, n, lams):
    errors=np.zeros((n,lams))
    for i in range(n):
        for lam in range(lams):
            F=computing_F(x, i)
            w=computing_w(F,t, lam)
            #new_y=F.dot(w)
            print w.shape
            #типа штуки с кросс-валидацией
            x1 = np.linspace(0, .1, 200)
            y = 20 * np.sin(2 * np.pi * 3 * x1) + 100 * np.exp(x1)
            e = 10 * np.random.randn(200)
            t1 = y + e
            #считаем по заданному
            new_y=computing_y(w, x1)
            errors[i,lam]=np.sum((new_y - t1) ** 2)
    find=np.where(errors == np.min(errors))
    print np.min(errors), np.where(errors==np.min(errors))
    return find[0], find[1]



def main():
    x = np.linspace(0, 1, 1000)
    y = 20 * np.sin(2 * np.pi * 3 * x) + 100 * np.exp(x)
    e = 10 * np.random.randn(1000)
    t = y + e
    plt.figure()
    plt.plot(x,t, '.g')
    plt.plot(x,y,'r')

    n,lams=find_min_error(x,t,30, 10)
    #на случай, если таких будет несколько
    for i in range(n.size):
        F = computing_F(x, n[i])
        w = computing_w(F, t, lams[i])
        new_y = F.dot(w)

    #plt.plot(np.arange(k), errors, 'g')

        plt.plot(x,new_y,color=(float(i)/n.size,0.,1. ))
    plt.show()

if __name__=="__main__":
    main()