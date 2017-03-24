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

def computing_new_y(w,x):
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

def find_min_error(training_x, training_set, valid_x, valid_set, ns, lams=np.zeros((1))):
    #все это надо переделать, но мне пока лень
    exp_count=ns.size*lams.size
    train_errors=np.zeros((exp_count))
    errors=np.zeros((exp_count))
    data=np.zeros((exp_count, 2))
    i=0
    for n in ns:
        for lam in lams:
    
            #n=i+1
            #n=i;
            #можно по-хитрому выщитывать лябда, но я решил, что при 0 - лучший вариант
            #lam=0#0np.random.uniform(0,100,1)
            F = computing_F(training_x, n)
            w = computing_w(F, training_set, lam)
            train_y=computing_new_y(w,training_x)
            valid_y=computing_new_y(w, valid_x)
            train_errors[i]=error(training_set, train_y,w,lam)
            errors[i]=error(valid_set, valid_y,w,lam)
            data[i][0]=n
            data[i][1]=lam
            i+=1
    plt.figure()
    plt.plot(data[:,0], train_errors,'g')
    plt.plot(data[:, 0], errors, 'b')
    plt.title('Error')
    plt.show()
    find=np.where(errors == np.min(errors))
    index=find[0][0]
    n=int(data[index][0])
    lam=data[index][1]
    F = computing_F(training_x, n)
    w = computing_w(F, training_set, lam)
    print( 'n=',n)
    print( 'lambda=',lam)
    #print w
    return w, lam

def error(x,y,w,lam,q=2):
    return (1/2.)*np.sum((x-y)**2)+(lam/2.)*np.sum(np.abs(w)**q)


def main():
    #Original
    x = np.linspace(0, 1, 2000)
    y = calculate_y(x)
    t=calculate_t(x)
    ar=np.arange(2000)
    np.random.shuffle(ar)

    # обучающая выборка
    training_x = x[ar[:1200]]
    training_set =  t[ar[:1200]]

    #валидационная выборка
    valid_x=x[ar[1200:1600]]
    valid_set = t[ar[1200:1600]]

    test_x=x[ar[1600:2000]]
    test_set=x[ar[1600:2000]]
    n=np.arange(30,100)
    lam=np.linspace(0,100,150)
    w,l = find_min_error(training_x, training_set, valid_x, valid_set,n, lam)

    training_new_y=computing_new_y(w, training_x)
    valid_new_y=computing_new_y(w, valid_x)
    test_new_y=computing_new_y(w, test_x)

    print ('Training-error: ', error(training_set, training_new_y,w,l))
    print ('Valid-error:    ', error(valid_set, valid_new_y,w,l))
    print ('Test-error:     ', error(test_new_y, test_set,w,l))
    new_x=np.linspace(0,1,500)
    new_y=computing_new_y(w,new_x)

    plt.figure()
    plt.plot(training_x, training_set, '.g')
    plt.plot(x, y, '.r')

    plt.plot(new_x, new_y, '.b')
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
