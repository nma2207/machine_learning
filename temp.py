#!/usr/bin/python
import numpy as np
import random
def main():
    n = 1
    m = 10
    k = 100
    N = 10000

    S = 0
    for i in range(N):
        kubs = np.zeros((n, m))
        s = 0
        for j in range(k):
            x = random.randint(0, n - 1)
            y = random.randint(0, m - 1)
            if (kubs[x, y] == 0):
                kubs[x, y] = 1
                s += 1
        #print s
        S += s
    print float(S) / N

    print "Hello, Python!"
if __name__=="__main__":
    main()