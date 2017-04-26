# -*- coding: utf-8 -*
import numpy as np
import matplotlib.pyplot as plt
import  math

BASIC_FUNC_COUNT=6
EPS=1
EPS0=1
def get_by_key(data, key):
    result=[]
    for i in data:
        result.append(i[key])
    return np.array(result)
def calc_F(x, y):
    #x=get_by_key(data, 'x')
    #y=get_by_key(data, 'y')
    # F = np.ones((x.size, 6))
    # F[:1]=x
    # F[:2]=x*y
    # F[:3]=x*y*y
    # F[:4]=np.sin(x)*np.cos(x)
    # F[:5]=y
    return np.array([x,x*y, x*y*y, math.sin(x)*math.cos(x),y, 1])

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
    #return data

def calc_E(t,x,y, w, lam):
    #result=0
    #for i in range (t.size):
    sigm=sigmoid(x,y,w)
    result=np.sum(t*np.log(sigm))
    result+=np.sum((1-t)*(1-np.log(sigm)))
    result+=lam/2*(np.sum(w.transpose().dot(w)))
    return result

def calc_gradient_E(t, x, y,w,lam):
    result=0.0
    for i in range(x.size):
        result+=calc_F(x[i],y[i])*(sigmoid(x[i],y[i], w)-t[i])
    result+=lam*w
    return result



def gradient_distent(data, gamma, lam):
    w=np.random.randn((BASIC_FUNC_COUNT))
    w_prev=np.copy(w)
    while(True):
        w_prev = np.copy(w)
        w=w_prev-gamma*calc_gradient_E(get_by_key(data, 'c'),
                                       get_by_key(data, 'x'),
                                       get_by_key(data, 'y'),
                                       w, lam)
        print 'error=',np.sum((w - w_prev) ** 2),EPS*(np.sum(w**2)+EPS0)
        if(np.sum((w-w_prev)**2)<EPS*(np.sum(w**2)+EPS0)):
            break
    return w

def find_best_w(train_data, valid_data):
    lams=np.linspace(1e-6,1,10)
    gamma=1
    best_w=np.zeros(BASIC_FUNC_COUNT)
    for lam in lams:
        w=gradient_distent(train_data, gamma, lam)
        print w



def sigmoid(x,y, w):
    #w=np.array(w)
    print np.sum(w.transpose().dot(calc_F(x, y)))
    result=1./(1+math.exp(np.sum(w.transpose().dot(calc_F(x, y)))))
    return result

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

    find_best_w(train_data, valid_data)

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