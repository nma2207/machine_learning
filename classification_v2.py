# -*- coding: utf-8 -*
import numpy as np
import matplotlib.pyplot as plt


def get_by_key(data, key):
    result=[]
    for i in data:
        result.append(i[key])
    return np.array(result)
def calc_F(x, y):
    #x=get_by_key(data, 'x')
    #y=get_by_key(data, 'y')
    F = np.ones((x.size, 6))
    F[:1]=x
    F[:2]=x*y
    F[:3]=x*y*y
    F[:4]=np.sin(x)*np.cos(x)
    F[:5]=y

def calc_y1(x):
    return 20 * np.sin(2 * np.pi * 3 * x) + 100 * np.exp(x)

def calc_y2(x):
    return 20 * np.sin(2 * np.pi * 3 * x) + 100 * np.exp(x)+50

def calc_t1(x):
    y=calc_y1(x)
    e = 5 * np.random.randn(x.size)
    t = y + e
    data=[]
    for i in range(x.size):
        data.append({'x':x[i],
                     'y':t[i],
                     'c':0})
    return np.array(data)

def calc_t2(x):
    y=calc_y2(x)
    e = 5 * np.random.randn(x.size)
    t = y + e
    data=[]
    for i in range(x.size):
        data.append({'x':x[i],
                     'y':t[i],
                     'c':1})
    return np.array(data)
    return data

def calc_E(t,x, w, lam):
    #result=0
    #for i in range (t.size):
    sigm=sigmoid(x,w)
    result=np.sum(t*np.log(sigm))
    result+=np.sum((1-t)*(1-np.log(sigm)))
    result+=lam/2*(np.sum(w.transpose().dot(w)))
    return result


def calc_w():
    print 'TODO'


def sigmoid(x, w):
    #TODO
    print 'TODO SIGMOID'

def main():
    data_count = 2000
    x=np.linspace(0,1,data_count)
    #np.random.shuffle(x)
    ar=np.arange(data_count)
    np.random.shuffle(ar)

    train_data=calc_t1(x[:(int(0.6*data_count))])
    train_data[:int(0.3*data_count)]=calc_t1(x[ar[:int(0.3*data_count)]])
    train_data[int(0.3 * data_count):int(0.6 * data_count)] = calc_t2(x[ar[int(0.3 * data_count):int(0.6 * data_count)]])

    valid_data=calc_t1(x[:(int(0.2*data_count))])
    valid_data[:int(0.1*data_count)]=calc_t1(x[ar[int(0.6*data_count):int(0.7*data_count)]])
    valid_data[int(0.1 * data_count):int(0.2 * data_count)] = calc_t2(x[ar[int(0.7 * data_count):int(0.8 * data_count)]])

    test_data=calc_t1(x[:(int(0.2*data_count))])
    test_data[:int(0.1*data_count)]=calc_t1(x[ar[int(0.8*data_count):int(0.9*data_count)]])
    test_data[int(0.1 * data_count):int(0.2 * data_count)] = calc_t2(x[ar[int(0.9 * data_count):int(1 * data_count)]])

    plt.figure()
    plt.plot(get_by_key(train_data,'x'), get_by_key(train_data, 'y'),'.g')
    plt.show()
    # d1=calc_t1(x)
    # d2=calc_t2(x)
    #plt.figure()
    #plt.plot(train_data['x'], train_data['y'],'.r')
    #plt.plot(d2['x'], d2['y'], '.b')
    #plt.show()

if __name__=="__main__":
    main()