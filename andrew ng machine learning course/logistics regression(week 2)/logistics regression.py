#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#importing dataset
data = pd.read_csv("ex2data2.txt",header = None)
data.head()
#seperating dependent and independent variables
x = data.iloc[:,:2]
y = data.iloc[:,-1]
#visualizing the re
#visualing
m = y == 1
adm = plt.scatter(x[m][0],x[m][1])
not_adm = plt.scatter(x[~m][0],x[~m][1])
plt.show()
from sklearn.linear_model import LogisticRegression
reg = LogisticRegression()
reg.fit(x,y)
y_pred = reg.predict(x)
print(y_pred)
#checking accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_pred,y)
print(accuracy)
#plotting the decision boudary
x_values = [np.min(x[:, 1] - 5), np.max(x[:, 2] + 5)]
y_values = - (parameters[0] + np.dot(parameters[1], x_values)) / parameters[2]
parameters = reg.coef_
parameters

plt.plot(x_values, y_values, label='Decision Boundary')
plt.xlabel('Marks in 1st Exam')
plt.ylabel('Marks in 2nd Exam')
plt.legend()
plt.show()
