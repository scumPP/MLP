import numpy as np
import matplotlib.pyplot as plt


def main():
    X_train=np.array([[3,3],[4,3],[1,1]])
    y_train=np.array([1,1,-1])

    perceptron=Perceptron()

    perceptron.fit(X_train,y_train)

    #draw(X_train,perceptron.w,perceptron.b)

'''
感知机具体实现方法
   感知机收敛条件：数据集必须是绝对线性可分
'''
class Perceptron:

    def __init__(self):
        self.w=None
        self.b=0
        self.l_rate=1  ##学习率

    def fit(self,X_train,y_train):
        ##为权重w设置初始值，根据训练样本x的特征个数进行设置
        self.w=np.zeros(X_train.shape[1])
        i=0
        while i<X_train.shape[0]:
            X=X_train[i]
            y=y_train[i]
            if y*(np.dot(self.w,X)+self.b)<=0:
                self.w=self.w+self.l_rate*np.dot(y,X)
                self.b=self.b+self.l_rate*y
                i=0
            else:
                i+=1

main()
