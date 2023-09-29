from sklearn.datasets import load_svmlight_file
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
class myPerceptron:
    def __init__(self,learning_rate=0.01, iters=10000):
        self.lr = learning_rate
        self.iters = iters
        self.activate_func = self.step_func
        self.weight=None
        self.bias =None

    def fit(self,X,y):
        assert X.shape[0] == y.shape[0]
        self.weight = np.zeros(X.shape[1])
        self.bias = 0
        y_ = np.where(y>0,1,0)
        self.best_acc = -1
        for _ in tqdm(range(self.iters),total=self.iters):
            
            for idx in range(X.shape[0]):
                x_i = X[idx,:]
                output = np.dot(x_i,self.weight) + self.bias
                predict_y = self.activate_func(output)

                update = self.lr * (y_[idx]-predict_y)
                self.weight += update * x_i
                self.bias += update

            tmp = self.accuracy(y,self.predict(X))
            if self.best_acc < tmp :
                self.best_acc = tmp
        print(f"best acc: {self.best_acc}")
    def sign(self,X):
        return np.where(X>=0,1,-1)

    def predict(self,X):
        output = X@self.weight.T+self.bias
        predict_y = self.sign(output)
        return predict_y

    def step_func(self,x):
        return np.where(x>=0,1,0)
    
    def accuracy(self,y_true,y_pred):
        return (y_true == y_pred).sum()/y_true.shape[0]
    
feats,labels = load_svmlight_file("diabetes_scale")
feats = np.asarray(feats.todense())

# cls = myPerceptron()
# cls.fit(feats,labels)
# pred = cls.predict(feats)
# print(f"acc: {cls.accuracy(labels,pred)}")

from sklearn.linear_model import Perceptron, LogisticRegression


cls = Perceptron()
cls.fit(feats,labels)
print(cls.score(feats,labels))

cls = LogisticRegression(penalty="l2", dual=True,solver="liblinear",C=1 ,max_iter=100000)
cls.fit(feats,labels)
print(cls.score(feats,labels))

from sklearn.svm import SVC, NuSVC
cls = SVC(C=1,kernel="linear")
cls.fit(feats,labels)
print(cls.score(feats,labels))

cls = SVC(C=1,kernel="poly")
cls.fit(feats,labels)
print(cls.score(feats,labels))

cls = SVC(C=1,kernel="sigmoid")
cls.fit(feats,labels)
print(cls.score(feats,labels))

cls = SVC(C=1)
cls.fit(feats,labels)
print(cls.score(feats,labels))

cls = NuSVC()
cls.fit(feats,labels)
print(cls.score(feats,labels))

from sklearn.neural_network import MLPClassifier

cls = MLPClassifier(max_iter=10000,activation="tanh",hidden_layer_sizes=(32,64,256,64,32))
cls.fit(feats,labels)
print(cls.score(feats,labels))

from sklearn.metrics import accuracy_score
y_true = labels
y_pred = np.ones(labels.shape)
y_pred_2 = np.zeros(labels.shape)-1

print(accuracy_score(y_true,y_pred))
print(accuracy_score(y_true,y_pred_2))

