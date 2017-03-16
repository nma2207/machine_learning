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

def calculate_y(x):
    return  20 * np.sin(2 * np.pi * 3 * x) + 100 * np.exp(x)

def calculate_t(x):
    y=calculate_y(x)
    e = 10 * np.random.randn(x.size)
    t = y + e
    return t

def find_min_error(training_x, training_set, valid_x, valid_set, exp_count):
    #все это надо переделать, но мне пока лень
    train_errors=np.zeros((exp_count))
    errors=np.zeros((exp_count))
    data=np.zeros((exp_count, 2))
    for i in range(exp_count):
        n=int(np.random.uniform(1,100,1))
        lam=np.random.uniform(-50,50,1)
        F = computing_F(training_x, n)
        w = computing_w(F, training_x, lam)
        train_y=computing_y(w,training_x)
        valid_y=computing_y(w, valid_x)
        train_errors[i]=np.sum((train_y-training_set)**2)
        errors[i]=np.sum((valid_y-valid_set)**2)
        data[i][0]=n
        data[i][1]=lam
    plt.figure()
    print data[:,0]
    plt.plot(data[:,0], train_errors,'g')
    plt.plot(data[:, 0], errors, 'b')
    plt.show()
    find=np.where(errors == np.min(errors))
    index=find[0][0]
    n=data[index][0]
    lam=data[index][1]
    print data[index]
    F = computing_F(training_x, n)
    w = computing_w(F, training_x, lam)
    return w



def main():
    #Original
    x = np.linspace(0, 1, 1000)
    y = calculate_y(x)

    # обучающая выборка
    training_x = np.random.uniform(0, 1, 1000)
    training_set = calculate_t(training_x)

    #валидационная выборка
    valid_x=np.random.uniform(0,1,200)
    valid_set = calculate_t(valid_x)

    test_x=np.random.uniform(0,1,200)
    test__set=calculate_t(test_x)

    w = find_min_error(training_x, training_set, valid_x, valid_set, 1000)

    new_x=np.linspace(0,1,500)
    new_y=computing_y(w,new_x)

    plt.figure()
    plt.plot(x, y, 'r')
    plt.plot(training_x,training_set, '.g')
    plt.plot(new_x, new_y, 'g')
    plt.show()



    #на случай, если таких будет несколько
    # for i in range(n.size):
    #     F = computing_F(x, n[i])
    #     w = computing_w(F, t, lams[i])
    #     new_y = F.dot(w)

    #plt.plot(np.arange(k), errors, 'g')

    # plt.plot(x,new_y,color=(float(i)/n.size,0.,1. ))
    # plt.show()

if __name__=="__main__":
    main()