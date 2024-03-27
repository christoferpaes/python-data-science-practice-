# -*- coding: utf-8 -*-
"""a8(1)(1).ipynb






## Data Scientists

Assume that we have a data-science club where data scientists meet and discuss data analysis and visualization. The members in the club are either paid accounts or unpaid accounts.  You are provided a list of tuples.  Each tuple contains three elements:
* tenure, which is the number of years as a data scientist, 
* salary, which is how much the data scientist ears, 
* account, which is a number that is either 1 for a paid account or 0 for an unpaid account.
"""

#do not change the below statement
data = [(0.7,48000,1),(1.9,48000,0),(2.5,60000,1),(4.2,63000,0),(6,76000,0),(6.5,69000,0),(7.5,76000,0),(8.1,88000,0),(8.7,83000,1),(10,83000,1),(0.8,43000,0),(1.8,60000,0),(10,79000,1),(6.1,76000,0),(1.4,50000,0),(9.1,92000,0),(5.8,75000,0),(5.2,69000,0),(1,56000,0),(6,67000,0),(4.9,74000,0),(6.4,63000,1),(6.2,82000,0),(3.3,58000,0),(9.3,90000,1),(5.5,57000,1),(9.1,102000,0),(2.4,54000,0),(8.2,65000,1),(5.3,82000,0),(9.8,107000,0),(1.8,64000,0),(0.6,46000,1),(0.8,48000,0),(8.6,84000,1),(0.6,45000,0),(0.5,30000,1),(7.3,89000,0),(2.5,48000,1),(5.6,76000,0),(7.4,77000,0),(2.7,56000,0),(0.7,48000,0),(1.2,42000,0),(0.2,32000,1),(4.7,56000,1),(2.8,44000,1),(7.6,78000,0),(1.1,63000,0),(8,79000,1),(2.7,56000,0),(6,52000,1),(4.6,56000,0),(2.5,51000,0),(5.7,71000,0),(2.9,65000,0),(1.1,33000,1),(3,62000,0),(4,71000,0),(2.4,61000,0),(7.5,75000,0),(9.7,81000,1),(3.2,62000,0),(7.9,88000,0),(4.7,44000,1),(2.5,55000,0),(1.6,41000,0),(6.7,64000,1),(6.9,66000,1),(7.9,78000,1),(8.1,102000,0),(5.3,48000,1),(8.5,66000,1),(0.2,56000,0),(6,69000,0),(7.5,77000,0),(8,86000,0),(4.4,68000,0),(4.9,75000,0),(1.5,60000,0),(2.2,50000,0),(3.4,49000,1),(4.2,70000,0),(7.7,98000,0),(8.2,85000,0),(5.4,88000,0),(0.1,46000,0),(1.5,37000,0),(6.3,86000,0),(3.7,57000,0),(8.4,85000,0),(2,42000,0),(5.8,69000,1),(2.7,64000,0),(3.1,63000,0),(1.9,48000,0),(10,72000,1),(0.2,45000,0),(8.6,95000,0),(1.5,64000,0),(9.8,95000,0),(5.3,65000,0),(7.5,80000,0),(9.9,91000,0),(9.7,50000,1),(2.8,68000,0),(3.6,58000,0),(3.9,74000,0),(4.4,76000,0),(2.5,49000,0),(7.2,81000,0),(5.2,60000,1),(2.4,62000,0),(8.9,94000,0),(2.4,63000,0),(6.8,69000,1),(6.5,77000,0),(7,86000,0),(9.4,94000,0),(7.8,72000,1),(0.2,53000,0),(10,97000,0),(5.5,65000,0),(7.7,71000,1),(8.1,66000,1),(9.8,91000,0),(8,84000,0),(2.7,55000,0),(2.8,62000,0),(9.4,79000,0),(2.5,57000,0),(7.4,70000,1),(2.1,47000,0),(5.3,62000,1),(6.3,79000,0),(6.8,58000,1),(5.7,80000,0),(2.2,61000,0),(4.8,62000,0),(3.7,64000,0),(4.1,85000,0),(2.3,51000,0),(3.5,58000,0),(0.9,43000,0),(0.9,54000,0),(4.5,74000,0),(6.5,55000,1),(4.1,41000,1),(7.1,73000,0),(1.1,66000,0),(9.1,81000,1),(8,69000,1),(7.3,72000,1),(3.3,50000,0),(3.9,58000,0),(2.6,49000,0),(1.6,78000,0),(0.7,56000,0),(2.1,36000,1),(7.5,90000,0),(4.8,59000,1),(8.9,95000,0),(6.2,72000,0),(6.3,63000,0),(9.1,100000,0),(7.3,61000,1),(5.6,74000,0),(0.5,66000,0),(1.1,59000,0),(5.1,61000,0),(6.2,70000,0),(6.6,56000,1),(6.3,76000,0),(6.5,78000,0),(5.1,59000,0),(9.5,74000,1),(4.5,64000,0),(2,54000,0),(1,52000,0),(4,69000,0),(6.5,76000,0),(3,60000,0),(4.5,63000,0),(7.8,70000,0),(3.9,60000,1),(0.8,51000,0),(4.2,78000,0),(1.1,54000,0),(6.2,60000,0),(2.9,59000,0),(2.1,52000,0),(8.2,87000,0),(4.8,73000,0),(2.2,42000,1),(9.1,98000,0),(6.5,84000,0),(6.9,73000,0),(5.1,72000,0),(9.1,69000,1),(9.8,79000,1),]

"""### Problem 1: Plotting the Club Members

For this problem, you need to plot the data scientists in the data-science club so that we can conveniently visualize their tenured years, salaries, and paid accounts or not. Note that this problem was approached in an earlier assignment. I expect to explore the data by approaching the question. 
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools

def plotClubMembers(x):
    data2 = []
    data3 = []
    data2x = []
    data2y = []
    data3x = []
    data3y = [] 
    datatx = []
    dataty = [] 
    dataframe = pd.DataFrame(data,columns=['Tenure', 'Salary', ' Account'])
    X = dataframe.iloc[:, :-1].values
    y = dataframe.iloc[:, -1].values
    zero = 0
    one = 1
    for i in data:
      datatx.append(i[0])
      dataty.append(i[1])

    for x in data:
        if zero in x:
            data2.append(x)
        
        else:
            data3.append(x)   

    for i in data2:
      data2x.append(i[0])
      data2y.append(i[2])
    for i in data3:
      data3x.append(i[0])
      data3y.append(i[2])
    labels=['Tenured', 'Nontenured']
    markers = ['X', 'd'] # setting markers
    colors = itertools.cycle(['green','red']) # setting colors
    #calling zip function
    plt.figure(figsize=(15, 10))

    plt.scatter(data3x, data3y,s=100,c=next(colors),marker= 'd', label = labels[0])

   
    plt.scatter(data2x, data2y,c = next(colors), marker = 'X', label = labels[1])

    #x = [2,4,6,8,10,12] #didnt need to set since the scatter plot function set the x 
    #y = [20000,40000,60000, 80000, 100000, 120000]  #didnt need to set since the scatter plot function set the y
    plt.xlabel('Years as a data scientist')
    plt.ylabel('Paid or unpaid')
    plt.title('Data Science Club Members')
    plt.legend(loc='upper left')
    plt.show()

def plotClubMembers1(x):
    data2 = []
    data3 = []
    data2x = []
    data2y = []
    data3x = []
    data3y = [] 
    datatx = []
    dataty = [] 
    dataframe = pd.DataFrame(data,columns=['Tenure', 'Salary', ' Account'])
    X = dataframe.iloc[:, :-1].values
    y = dataframe.iloc[:, -1].values
    
    zero = 0
    one = 1
    for i in data:
      datatx.append(i[0])
      dataty.append(i[1])

    for x in data:
        if zero in x:
            data2.append(x)
        
        else:
            data3.append(x)   

    for i in data2:
      data2x.append(i[0])
      data2y.append(i[1])
    for i in data3:
      data3x.append(i[0])
      data3y.append(i[1])
    labels=['Tenured', 'Nontenured']
    markers = ['X', 'd'] # setting markers
    colors = itertools.cycle(['green','red']) # setting colors
    #calling zip function
    plt.figure(figsize=(15, 10))

    plt.scatter(data3x, data3y,s=100,c=next(colors),marker= 'd', label = labels[0])

   
    plt.scatter(data2x, data2y,c = next(colors), marker = 'X', label = labels[1])

    #x = [2,4,6,8,10,12] #didnt need to set since the scatter plot function set the x 
    #y = [20000,40000,60000, 80000, 100000, 120000]  #didnt need to set since the scatter plot function set the y
    plt.xlabel('Years as a data scientist')
    plt.ylabel('Salary')
    plt.title('Data Science Club Members')
    plt.legend(loc='upper left')
    plt.show()

plotClubMembers(data) ### showing how we can turn this dataset into a binary analysis for logistic regression

plotClubMembers1(data) ### visualization of plotted data science club memebers that are split between paid and unpaid

"""### Problem 2  Preparing for Building Learning Algorithms 

For this problem, you need to write functions/class definition(s) to prepare building classifiers and testing them using various metrics including accuracy, sensitivity, specificity, positive predicative value, negative predictive value.  In addition, you need to define a class DataScientist from which you can create data examples.  Each example represents a data scientist in terms of its features including salary and tenured years.  Your class definition also needs to allow you label a data scientist appropriately as paid or unpaid account in the club. 

After you complete your class DataScientist definition, you need to process the provided data into a list of data examples. (Each example is a data scientist.)  

Note that you can reuse as much code as possible from the lecture notes.  Indeed, I expect to reuse a lot of code from the lecture notes to approach the problems in this assignment. 
"""

dataframe = pd.DataFrame(data,columns=['Tenure', 'Salary', ' Account']) ## must be represented in a dataframe to train model
X = dataframe.iloc[:, :-1].values 
y = dataframe.iloc[:, -1].values

print(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state = 0)

print(type(y_train))

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

"""### Problem 3 Logistic Regression 

In this problem, you need to write function(s) to build a classifier using logistic regression algorithm.  Then, you need to apply the test methods (leaveOneOut and randomSplit) to evaluate the learned classifier in terms of accuracy, sensitivity, specificity, and positive predicative value.

The output of your evaluation on the logistic-regression classifier you built could be similar to the below:
```
Average of 10 80/20 splits LR
 Accuracy = 0.858
 Sensitivity = 0.614
 Specificity = 0.94
 Pos. Pred. Val. = 0.775
 
Average of LOO testing using LR
 Accuracy = 0.875
 Sensitivity = 0.635
 Specificity = 0.959
 Pos. Pred. Val. = 0.846
 ```
 
 Additionally, you need to plot the ROC curve and compute the AUC score to evaluate your classifier.  
"""

from matplotlib.colors import ListedColormap
plt.figure(figsize=(20,10))
X_set, y_set = sc.inverse_transform(X_train), y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0 ].max() +10, step =0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() +1000, step = 0.25) )
plt.contourf(X1,X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
                      alpha= 0.75, cmap = ListedColormap(('blue', 'gold')))
plt.xlim(X1.min(), X1.max() )
plt.ylim(X2.min(), X2.max() )
for i, j in enumerate(np.unique(y_set)):
  plt.scatter(X_set[ y_set == j , 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green')) (i), label= j      )
plt.title('Logistic Regression Training Set')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
data = pd.DataFrame(y_train,columns=['outcome'])
data.outcome[data.outcome == 1] = 'Tenured'
data.outcome[data.outcome == 0] = 'Non Tenured'
print(pd.DataFrame(X_set, columns=['Tenure', 'Salary']),data)
print(range(len(data.outcome[data.outcome == 'Tenured'])))
print(range(len(data.outcome[data.outcome == 'Non Tenured'])))

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

"""### Problem 4 KNN

In this problem, you need to write function(s) to build a classifier using KNN algorithm.  Then, you need to apply the test methods (leaveOneOut and randomSplit) to evaluate the learned classifer in terms of accuracy, sensitivity, specificity, and positive predicative value.

The output of your evaluation on the KNN classifier you built could be similar to the below:
```
Average of 10 80/20 splits using KNN (k=3)
 Accuracy = 0.857
 Sensitivity = 0.634
 Specificity = 0.933
 Pos. Pred. Val. = 0.762
 
Average of LOO testing using KNN (k=3)
 Accuracy = 0.85
 Sensitivity = 0.615
 Specificity = 0.932
 Pos. Pred. Val. = 0.762
 ```
"""

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 3)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

from matplotlib.colors import ListedColormap
plt.figure(figsize=(20,10))
X_set, y_set = sc.inverse_transform(X_train), y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 1),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 1))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('K-NN (Training set)')
plt.xlabel('Tenure')
plt.ylabel('Salary')
plt.legend()
plt.show()
data = pd.DataFrame(y_train,columns=['outcome'])
data.outcome[data.outcome == 1] = 'Tenured'
data.outcome[data.outcome == 0] = 'Non Tenured'
print(pd.DataFrame(X_set, columns=['Tenure', 'Salary']),data)
print(range(len(data.outcome[data.outcome == 'Tenured'])))
print(range(len(data.outcome[data.outcome == 'Non Tenured'])))

"""## Loan

In the following problems, you will analyze a set of loan data points.  Each data point is presented as a row in the data file (loan_data.csv).   Each row contains the customer data including id, outcome, dti, borrower_score, and payment_inc_ratio. The loan outcome should be used to label the data point for your classifier.  I would suggest you not use id as a feature for your feature vectors.

### Problem 5  Preparing for Building Learning Algorithms 

For this problem, you need to write functions/class definition(s) to prepare building classifiers and testing them using various metrics including accuracy, sensitivity, specificity, positive predicative value, negative predictive value. In Problem 2, you are also required to define the metric functions. So, you can reuse the functions you defined there. In addition, you need to define a class Customer from which you can create data examples.  Each example represents a Customer in terms of its features.  Your class definition also needs to allow you label a customer appropriately based on the loan outcome "paid off" or "default". 

After you complete your class Customer definition, you need to process the provided data into a list of data examples. (Each example is a customer.)  

Again, I would emphasize that you can reuse as much code as possible from the lecture notes and share your code to solve the problems in this assignment.
"""

def getData(fileName):
    dataFile = open(fileName, 'r')
    id = []
    outcome = []
    dti = []
    borrower_score = []
    payment_inc_ratio = [] 

    dataFile.readline()
    for line in dataFile:
        i,g,a,c,s = line.split(',')
        id.append(float(i))
        outcome.append(str(g))
        dti.append(float(a))
        borrower_score.append(float(c))
        payment_inc_ratio.append(float(s))
    dataFile.close()
    return (outcome,dti, borrower_score, payment_inc_ratio)

# import pandas library 
import pandas as pd 
  
# creating file handler for 
# our example.csv file in 
# read mode 
file_handler = open('loan_data.csv', "r") 
  
# creating a Pandas DataFrame 
# using read_csv function that 
# reads from a csv file. 
data = pd.read_csv(file_handler, sep = ",") 
  
# closing the file handler 
file_handler.close() 
  
# traversing through Gender  
# column of dataFrame and  
# writing values where 
# condition matches. 
data.outcome[data.outcome == 'paid off'] = 1
data.outcome[data.outcome == 'default'] = 0
print(data)
#print(data.id, data.outcome,data.dti, data.borrower_score,data.payment_inc_ratio)

dataframe = dataframe.reset_index()
dataframe = pd.DataFrame(data,columns=['dti','payment_inc_ratio' ,'outcome']) ## must be represented in a dataframe to train model

X = dataframe.iloc[:, :-1].values 
y = dataframe.iloc[:, -1].values
y=y.astype('int') ### solution to https://stackoverflow.com/questions/45346550/valueerror-unknown-label-type-unknown

print(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print(X_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(X_test)

"""### Problem 6 Logistic Regression 

In this problem, you need to write function(s) to build a classifier using logistic regression algorithm.  Then, you need to apply the test methods (leaveOneOut and randomSplit) to evaluate the learned classifier in terms of accuracy, sensitivity, specificity, and positive predicative value.

 Additionally, you need to plot the ROC curve and compute the AUC score to evaluate your classifier.  
"""

from matplotlib.colors import ListedColormap

X_set, y_set = sc.inverse_transform(X_train), y_train

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0 ].max() +10, step =0.25),
                       
                    np.arange(start = X_set[:, 1].min() - 10, stop = X_set[:, 1].max() +10, step = 0.25) )
  
plt.contourf(X1,X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
                      alpha= 0.75, cmap = ListedColormap(('blue', 'gold')))
plt.xlim(X1.min(), X1.max() )
plt.ylim(X2.min(), X2.max() )
for i, j in enumerate(np.unique(y_set)):
  plt.scatter(X_set[ y_set == j , 0], X_set[y_set == j, 1], c = ListedColormap(('gold', 'blue')) (i), label= j      )
plt.title('Logistic Regression Training Set')
plt.xlabel('Income Ratio')
plt.ylabel('salary')
plt.legend()
plt.show()
data = pd.DataFrame(y_train,columns=['outcome'])
data.outcome[data.outcome == 1] = 'paid off'
data.outcome[data.outcome == 0] = 'default'

  
print(pd.DataFrame(X_set, columns=['income ratio', 'salary ']),data)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

"""## Problem 7 Summary Writeup

For this problem, you are expected to reflect on the classifiers you built on the two data sets (data scientists and loan customers).  You need to address the below questions:

* While building the classifiers using KNN and logistic regression, how did you build your examples to train and test your classifiers?

* How do you think the classifiers? Any one is better? And why?
"""

#### OBSERVATIONS FROM TESTING ####
 """
#### Observations from Testing ####

Within the first three problems, we had a tuple list of data given to build our models with, which indicated whether individuals were tenured or non-tenured (0 or 1). Using KNN, we can observe the majority of cases for tenured and non-tenured individuals, as well as which feature vectors have the greatest effect on the prediction.

The majority of predictions for the cases would be non-tenured (117) versus 43 tenured cases, accounting for 26.875% of tenured cases. The majority of cases for tenured individuals are related to the feature vector concerning the number of years, with tenured amounts of 10 years or greater and salaries ranging from $70,000 to $80,000.

While the majority of non-tenured professors do have higher incomes, their population remains at 73.125%, and they have higher-income earners. Additionally, nearly 50% more of the population is non-tenured. However, there are obviously trade-offs between being non-tenured and a tenured professor.
OBSERVATIONS FROM TESTING
Data and Classifiers

The initial problems involved building classifiers using a tuple list of data scientists' tenure status (tenured: 1, non-tenured: 0) and salary. We employed K-Nearest Neighbors (KNN) to analyze the data, specifically focusing on:

Tenure Distribution: KNN allows us to observe the majority class (likely non-tenured in this case) and identify feature vectors (like years of experience and salary) that significantly influence tenure predictions.
Tenure Prediction: Based on the provided data, KNN might predict a majority of non-tenured cases (e.g., 117 vs. 43 tenured, representing 26.875% tenured).
Observations on Tenure and Salary

Tenured Professors: Tenure might be more likely for individuals with:
10 years or more of experience.
Salaries ranging from $70,000 to $80,000 (though this may not be the sole determining factor).
Non-Tenured Professors: While some non-tenured professors might have higher salaries, they likely make up a larger portion of the population (e.g., 73.125%).
Considerations and Trade-offs

These observations suggest a potential correlation between tenure, experience, and salary. However, it's crucial to remember that these are just initial findings based on a limited dataset. Tenure decisions often involve a multifaceted evaluation encompassing not just years of service and salary but also research contributions, teaching effectiveness, and service to the university community.

Therefore, it's important to acknowledge the trade-offs between tenured and non-tenured positions:

Tenure: Offers job security and academic freedom but may come with greater research and service expectations.
Non-Tenure: May provide more flexibility in teaching focus but lacks the job security of tenure.
Future Exploration

Further investigation with a more comprehensive dataset could provide deeper insights into the relationships between tenure, experience, salary, and other relevant factors. Additionally, exploring other classification algorithms (like Logistic Regression) might yield different perspectives on the data.

By critically evaluating these findings and conducting more comprehensive analyses, we can gain a more nuanced understanding of tenure and its potential links to experience and salary in academia.

"""

