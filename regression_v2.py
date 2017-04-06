# coding:utf-8
import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg as nalg
import random
import math
#  ВНИМАНИЕ!!!!
#  ИСПОЛЬЗУЕТСЯ Python 2.7

# n - степень полинома
def computing_F(x, n):
    F = np.ones((x.size, n + 6))
    for i in range(1, n + 1):
        F[:, i] = x ** i
    F[:,n+1]=np.cos(x)
    F[:,n+2]=np.sin(x)
    #F[:,n+3]=np.log(x+1)
    F[:,n+3]=np.sqrt(x)
    F[:,n+4]=np.exp(x)
    F#[:,n+6]=np.exp(-x)
    F[:,n+5]=np.tan(x);
    return F


def computing_w(F, t, lam=0):
    I = np.identity(F.shape[1])
    return nalg.inv((F.transpose().dot(F)) + lam * I).dot(F.transpose()).dot(t)


def computing_new_y(w, x):
    y = np.zeros((x.size), dtype=float)
    for i in range(w.size):
        y += w[i] * (x ** i)
    return y


def calculate_y(x):
    return 20 * np.sin(2 * np.pi * 3 * x) + 100 * np.exp(x)


def calculate_t(x):
    y = calculate_y(x)
    e = 10 * np.random.randn(x.size)
    t = y + e
    return t


def find_min_error(training_x, training_set, valid_x, valid_set, ns, lams=np.zeros((1))):
    # все это надо переделать, но мне пока лень
    exp_count = ns * lams.size
    train_errors = np.zeros((exp_count))
    errors = np.zeros((exp_count))
    data = np.zeros((exp_count, 2))
    global_F=computing_F(training_x, 10)
    i = 0
    good_w=np.zeros((1))
    good_lam=0
    good_finctions=[]
    min_error=-1
    for n in range(ns):
        for lam in lams:
            # n=i+1
            # n=i;
            # можно по-хитрому выщитывать лябда, но я решил, что при 0 - лучший вариант
            # lam=0#0np.random.uniform(0,100,1)
            f_count=np.arange(global_F.shape[1])
            np.random.shuffle(f_count)
            f_count=f_count[:random.randint(1, f_count.size-1)]
            F=global_F[:,f_count]
            #print 'f count',f_count.size
            #print
            w = computing_w(F, training_set, lam)
            train_y = computing_new_y(w, training_x)
            valid_y = computing_new_y(w, valid_x)
            train_errors[i] = error(training_set, train_y, w, lam)
            errors[i] = error(valid_set, valid_y, w, lam)
            data[i][0] = n
            data[i][1] = lam

            if(errors[i]<min_error or min_error==-1):
                min_error=errors[i]
                good_w=w
                good_lam=lam
                good_finctions=f_count
            i += 1

    plt.figure()
    plt.plot(data[:, 0], train_errors, 'g', label='training')
    plt.plot(data[:, 0], errors, 'b', label='valid')
    plt.legend()
    plt.title('Error')
    plt.show()
    print 'good func=', good_finctions
    #print('n=', n)
    print('lambda=', good_lam)
    # print w
    return good_w, good_lam


def error(x, y, w, lam, q=2):
    #решил разделить на сумму, т.к. нечестно не делить
    #для training set ошибка получалась больше во столько раз,
    #  сколько там элементов. можно лучше даже написать как среднеквадратическое отклонение
    return ((1 / 2.) * np.sum((x - y) ** 2)/x.size + (lam / 2.) * np.sum(np.abs(w) ** q))


def main():
    # Original
    data_count=1000
    x = np.linspace(0, 1, data_count)
    y = calculate_y(x)
    t = calculate_t(x)
    ar = np.arange(data_count)
    np.random.shuffle(ar)

    # обучающая выборка
    training_x = x[ar[:int(0.6*data_count)]]
    training_set = t[ar[:int(0.6*data_count)]]

    # валидационная выборка
    valid_x = x[ar[int(0.6*data_count):int(0.8*data_count)]]
    valid_set = t[ar[int(0.6*data_count):int(0.8*data_count)]]

    test_x = x[ar[int(0.8*data_count):data_count]]
    test_set = t[ar[int(0.8*data_count):data_count]]

    lam = np.linspace(0, 1e-4, 100)

    w, l = find_min_error(training_x, training_set, valid_x, valid_set, 1000, lam)

    training_new_y = computing_new_y(w, training_x)
    valid_new_y = computing_new_y(w, valid_x)
    test_new_y = computing_new_y(w, test_x)

    print ('Training-error: ', error(training_set, training_new_y, w, l))
    print ('Valid-error:    ', error(valid_set, valid_new_y, w, l))
    print ('Test-error:     ', error(test_new_y, test_set, w, l))
    new_x = np.linspace(0, 1, 500)
    new_y = computing_new_y(w, new_x)

    plt.figure()
    plt.plot(training_x, training_set, '.g',label='training')
    plt.plot(valid_x, valid_set,'.y',label='valid')
    plt.plot(test_x, test_set, '.k',label='test')
    #plt.plot(x, y, '.r', label='real values')


    plt.plot(new_x, new_y, '.b', label='our values')
    plt.legend()
    plt.show()



if __name__ == "__main__":
    main()
