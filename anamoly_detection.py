from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.metrics import f1_score


data = loadmat('ex8data1.mat')
print data.keys()
X = data['X']


plt.scatter(X[:,0],X[:,1])
#plt.show()


mu = X.mean(axis=0)
sigma = X.var(axis=0)


#finding product of guassion distributions of dataset
prob = np.zeros((X.shape[0], X.shape[1]))
prob[:,0] = norm(mu[0], sigma[0]).pdf(X[:,0])
prob[:,1] = norm(mu[1], sigma[1]).pdf(X[:,1])
prob_product = np.multiply(prob[:, 0], prob[:, 1])

#cross_validation dataset
cv_X = data['Xval']
cv_y = data['yval']

#finding product of guassion distributions of cross_validation dataset
cv_prob = np.zeros((cv_X.shape[0], cv_X.shape[1]))
cv_prob[:, 0] = norm(mu[0], sigma[0]).pdf(cv_X[:, 0])
cv_prob[:, 1] = norm(mu[1], sigma[1]).pdf(cv_X[:, 1])

cv_prob_product = np.multiply(cv_prob[:, 0], cv_prob[:, 1])

#findind epsilon and f1_score
best_epsilon = 0
f1_max = 0
f = 0
step = (cv_prob_product.max()- cv_prob_product.min())/1000

for epsilon in np.arange(cv_prob_product.min(), cv_prob_product.max(), step):
    pred = (cv_prob_product < epsilon)

    f = f1_score(cv_y, pred, average='micro')




    if f > f1_max:
        f1_max = f
        best_epsilon = epsilon

print f1_max, best_epsilon
outliers = np.where(prob_product < best_epsilon)
print outliers


plt.scatter(X[outliers,0],X[outliers,1],color = "r")
plt.show()
