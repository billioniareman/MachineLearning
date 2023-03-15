# importing all required modules and library
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
# taking new data into a 2D array format
print("enter your samples dimensions in the sequence[petal length, petal width, sepal length,sepal width] all in cm:")
a = [ 'sepal length', 'sepal width','petal length', 'petal width']
b = []
for i in a:
    new = float(input('{}:'.format(i)))
    b.append(new)
newdata = np.array(np.array(b).reshape(-1, 1)).reshape(1, -1)
#training the model to classify between categories
knn = KNeighborsClassifier()
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], random_state=0)
knn.fit(X_train, y_train)
# making prediction of new data
predict = knn.predict(newdata)
print('the iris sample is of {} category'.format(''.join(iris['target_names'][predict]).upper()))
print('our prediction is {:.2f} % correct'.format(knn.score(X_test, y_test) * 100))
