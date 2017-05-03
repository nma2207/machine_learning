# -*- coding: utf-8 -*
import numpy as np
import numpy.linalg as nalg
import matplotlib.pyplot as plt
import  math

#
#определим некоторые важные константы
#
BASIC_FUNC_COUNT=5
EPS=1e-2
EPS0=1e-1


#
#Параметры для оценки классификатора
#
def calcilate_parametrs(result, real_value):
    tp=0
    tn=0
    fp=0
    fn=0
    for i in range(result.size):
        if(result[i]<=0.5 and real_value[i]==0):
            tn+=1
        elif (result[i]>0.5 and real_value[i]==0):
            fn+=1
        elif (result[i]<=0.5 and real_value[i]==1):
            fp+=1
        elif (result[i]>0.5 and real_value[i]==1):
            tp+=1
    precision=float(tp)/(tp+fp)
    recal=float(tp)/(tp+fn)
    params={
            'TP':  tp,
            'TN':tn,
            'FP':fp,
            'FN':fn,
            #'alfa': float(fp)/(tn+fp),
            #'betta':float(fn)/(tp+fn),
            'accuracy': float(tp+tn)/result.size,
            'precision':precision,
            'recal':recal,
            'f(1-score)':2.*(precision*recal)/(precision+recal)
    }
    return params

#
#Для получения массивов данных по ключу
#
def get_by_key(data, key):
    result=[]
    for i in data:
        result.append(i[key])
    return np.array(result)

#
#Наши базисные функции
#
def calc_F(x, y):
    return np.array([x,y, 0.1, x*y, x+y])

def calc_y1(x):
    return 20 * np.sin(2 * np.pi * 3 * x) + 100 * np.exp(x)

def calc_y2(x):
    return 20 * np.sin(2 * np.pi * 3 * x) + 100 * np.exp(x)+50

def calc_t1(x):
    y=calc_y1(x)
    e = 5 * np.random.randn(x.size)
    t = y + e
    t/=300
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
    t/=300
    data=[]
    for i in range(x.size):
        data.append({'x':x[i],
                     'y':t[i],
                     'c':1})
    return np.array(data)
    #return data


#
#то что мы хотим минимизировать
#
def calc_E(t,x,y, w, lam):
    sigm=sigmoid_narr(x,y,w)
    #print sigm[:10]
    sigm=np.array(sigm)
    result=np.sum(t*np.log(sigm))
    result+=np.sum((1-t)*(1-np.log(sigm)))
    #print 'e',result
    result*=-1
    result+=lam/2*(np.sum(w*w))
    return result

#
#Градиент того, что мы хотим минимизоровать
#
def calc_gradient_E(t, x, y,w,lam):
    result=0.0
    for i in range(x.size):
        result+=calc_F(x[i],y[i])*(sigmoid(x[i],y[i], w)-t[i])
    result+=lam*w
    return result


#
#Градиентный спуск
#
def gradient_distent(data, gamma, lam):
    w=np.ones((BASIC_FUNC_COUNT))
    w_prev=np.copy(w)
    errors=[]
    w_arr=[]
    while(True):
        w_prev = np.copy(w)
        grad=calc_gradient_E(get_by_key(data, 'c'),
                                       get_by_key(data, 'x'),
                                       get_by_key(data, 'y'),
                                       w, lam)

        w=w_prev-gamma*grad
        err=calc_E(get_by_key(data, 'c'),
                   get_by_key(data, 'x'),
                   get_by_key(data, 'y'),
                   w, lam)
        print nalg.norm(w-w_prev),'<?',EPS*(nalg.norm(w)+EPS0)
        errors.append(err)
        if(nalg.norm(w-w_prev)<EPS*(nalg.norm(w)+EPS0)):
            break
    errors=np.array(errors)
    print errors.size
    print type(np.arange(errors.size))
    plt.figure()
    plt.plot(np.arange(errors.size), errors, '.')
    plt.show()
    return w

#
#в поисках лучшего
#
def find_best_w(train_data, valid_data):
    lams=np.linspace(0,1e-6,10)
    gamma=0.01
    best_w=np.zeros(BASIC_FUNC_COUNT)
    errors=[]
    w_arr=[]
    for lam in lams:
        w=gradient_distent(train_data, gamma, lam)
        error=calc_E(get_by_key(valid_data, 'c'),
                     get_by_key(valid_data, 'x'),
                     get_by_key(valid_data, 'y'),
                     w, lam)
        errors.append(error)
        w_arr.append(w)
    errors=np.array(errors)
    min_i=np.where(errors==np.min(errors))
    min_i=min_i[0][0]
    print min_i
    print errors
    print lam
    return w_arr[min_i]


def sigmoid(x,y, w):
    result=1./(1+math.exp(-np.sum(w*(calc_F(x, y)))))
    return result

#
#Когда нам нужны сигмоиды для кучи данных
#
def sigmoid_narr(x,y,w):
    sigm=[]
    for i in range(x.size):
        sigm.append(sigmoid(x[i], y[i], w))

    sigm=np.array(sigm)
    return sigm

def main():
    data_count = 2000
    x=np.linspace(0,1,data_count)
    #np.random.shuffle(x)
    ar=np.arange(data_count)
    np.random.shuffle(ar)

    #
    #Тут очень хитро генерируем данные
    #
    train_data=calc_t1(x[:(int(0.6*data_count))])
    train_data[:int(0.3*data_count)]=calc_t1(x[ar[:int(0.3*data_count)]])
    train_data[int(0.3 * data_count):int(0.6 * data_count)] = calc_t2(x[ar[int(0.3 * data_count):int(0.6 * data_count)]])
    #print 'print max',np.max(get_by_key(train_data, 'y'))
    plt.figure()
    plt.plot(get_by_key(train_data, 'x'),
             get_by_key(train_data, 'y'),
             '.')
    plt.show()
    valid_data=calc_t1(x[:(int(0.2*data_count))])
    valid_data[:int(0.1*data_count)]=calc_t1(x[ar[int(0.6*data_count):int(0.7*data_count)]])
    valid_data[int(0.1 * data_count):int(0.2 * data_count)] = calc_t2(x[ar[int(0.7 * data_count):int(0.8 * data_count)]])

    test_data=calc_t1(x[:(int(0.2*data_count))])
    test_data[:int(0.1*data_count)]=calc_t1(x[ar[int(0.8*data_count):int(0.9*data_count)]])
    test_data[int(0.1 * data_count):int(0.2 * data_count)] = calc_t2(x[ar[int(0.9 * data_count):int(1 * data_count)]])

    w=find_best_w(train_data, valid_data)

    result=sigmoid_narr(get_by_key(test_data, 'x'),
                        get_by_key(test_data, 'y'),
                        w)
   # print result
    params=calcilate_parametrs(result, get_by_key(test_data, 'c'))
    print params
    #print get_by_key(test_data, 'c')
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