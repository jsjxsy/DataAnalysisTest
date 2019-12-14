# encoding=utf-8
import numpy as np

def startMain():
    lst = [[1,3,5],[2,4,6]]
    print (type(list))
    np_list = np.array(lst)
    print (type(np_list))
    np_list = np.array(lst, dtype=np.float)
    print (type(np_list))
    print (np_list.shape)
    print (np_list.ndim)
    print (np_list.dtype)
    print (np_list.itemsize)
    print (np_list.size)

    #some arrays
    print (np.zeros([2,4]))
    print (np.ones([3,5]))
    print (np.random.rand(2,4))
    print (np.random.rand())
    print (np.random.randint(1,10,3))
    print (np.random.randn(2,4))
    print (np.random.choice([10,20,30]))
    print (np.random.beta(1,10,100))


    #array opertation
    print (np.arange(1,11).reshape([2,-1]))
    lst = np.arange(1,11).reshape([2,-1])
    print (np.exp(lst))
    print (np.exp2(lst))
    print (np.sqrt(lst))
    print (np.sin(lst))
    print (np.log(lst))

    lst = np.array([])


    from numpy.linalg import *
    print ('linalg')
    print (np.eye(3))


    #others
    print ('FFT:')
    print (np.fft.fft(np.array([1,1,1,1,1,1,1])))
    print('poly')
    pass


if __name__ == '__main__':
    startMain()
