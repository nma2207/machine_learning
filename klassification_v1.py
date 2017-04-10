# -*- coding: utf-8 -*
import numpy as np
import matplotlib.pyplot as plt

#
#ВнИМАНиЕ!111
#используес python 2.7
def calcilate_parametrs(result):
    tp=0
    tn=0
    fp=0
    fn=0
    for i in range(len(result)//2):
        #True Negative
        if result[i]==0:
            tn+=1
        #Fasle Negative
        else:
            fn+=1
    for i in range(len(result)//2, len(result)):
        if(result[i]==1):
            tp+=1
        else:
            fp+=1
    precision=float(tp)/(tp+fp)
    recal=float(tp)/(tp+fn)
    params={
            'TP':  tp,
            'TN':tn,
            'FP':fp,
            'FN':fn,
            'alfa': float(fp)/(tn+fp),
            'betta':float(fn)/(tp+fn),
            'accuracy': float(tp+tn)/result.size,
            'precision':precision,
            'recal':recal,
            'f(1-score)':2.*(precision*recal)/(precision+recal)}
    return params
    
def classifier_1(mens):
    result=np.random.randint(0,2, mens.size)
    #футболитсы  - negative
    #баскетболиты - positive
    return result

def classifier_2(mens, thresh=190):
    result=[]
    for men in mens:
        if men<thresh:
            result.append(0)
        else:
            result.append(1)
    return np.array(result)
    
def main():
    mens=np.ones((1000))
    mens[0:500]*=180 #футболитсы
    mens[500:1000]*=200 #баскетболитсыц
    noise=np.random.randn(mens.size)*15
    mens+=noise
    res1=classifier_1(mens)
    print'res1=',calcilate_parametrs(res1)
    print
    
    res2=classifier_2(mens)
    print'res2=', calcilate_parametrs(res2)
    print
    #порог = 185
    res3=classifier_2(mens, 185)
    print'Thresh=185', calcilate_parametrs(res3)
    print
    #порог = 195
    res4=classifier_2(mens, 195)
    print'Thresh=195', calcilate_parametrs(res4)
    alfs=[]
    betts=[]
    for i in np.arange(180, 200, 0.05):
        res=classifier_2(mens, i)
        par=calcilate_parametrs(res)
        alfs.append( par['alfa'])
        betts.append(1- par['betta'])
    alfs=np.array(alfs)
    betts=np.array(betts)
    plt.figure()
    plt.plot(alfs, betts, '.b')
    plt.show()
if __name__=="__main__":
    main()