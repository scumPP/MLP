from sklearn.linear_model import Perceptron
import numpy as np

##训练数据集
X_train=np.array([[3,3],[4,3],[1,1]])
y=np.array([1,1,-1])
##构建perceptron对象，训练数据并输出结果
perceptron=Perceptron(tol=1e-3,eta0=1.0,penalty='elasticnet')
perceptron.fit(X_train,y)
print('w:',perceptron.coef_,'\n','b:',perceptron.intercept_,'\n','n_iter:',perceptron.n_iter_)
##测试模型预测的准确率
res=perceptron.score(X_train,y)
print('correct rate:{:.0%}'.format(res))
