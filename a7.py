# -*- coding: utf-8 -*-
"""a7.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1tahVlS5KlI4JzkIGg9FwNcuUgdqfmDpm

# Assignment 7 Clustering


## Learning Objectives

* Identify clustering learning algorithms
* Identify what is K-means clustering and how it works
* Apply K-means to build data models
* Analyze and communicate analysis results by applying K-means to learn from data

# Meet Ups

Assume that we want to celebrate our accomplishments on data analysis and visualization.  In the following three problems, you need to organize several in-person meet-ups for your hometown classmates.   You know the locations of all your local classmates, which are specified a list of lists in the below code cell.  Each inner list is a pair,

$[x,y]$

where $x$ stands for the number of blocks east of your home city center and $y$ stands for the number of blocks north of the city center.

That is, if x = -14, the location is located 14 blocks west of the home city; if y = 13, the location is located 13 blocks north of the home city.  If [x,y] is [-14, -5], which is the first location on the below list, it means the location is 14 blocks west and 5 blocks south of the home city center.    

You need to group the local classmates appropriately and choose meetup locations for everyone to attend conveniently.
"""

#do not change the blow statement
inputs = [[-14,-5],[13,13],[20,23],[-19,-11],[-9,-16],[21,27],[-49,15],[26,13],[-46,5],[-34,-1],[11,15],
          [-49,0],[-22,-16],[19,28],[-12,-8],[-13,-19],[-41,8],[-11,-6],[-25,-9],[-18,-3]]

"""## Problem 1: Plotting the Locations

For this problem, you need to plot the locations of the local classmates so that your local classmates' locations could be visualized easily.   
"""

def minkowskiDist(v1, v2, p):
    #Assumes v1 and v2 are equal length arrays of numbers
    dist = 0
    for i in range(len(v1)):
        dist += abs(v1[i] - v2[i])**p
    return dist**(1/p)


def distanceformula(x2,x1,y1,y2):
    distance = 0
    for k in range(len(x1)):
        y3 = (x2-x1)**2 + (y2-y1)**2
        distance += math.sqrt(y3)
        return distance

# Commented out IPython magic to ensure Python compatibility.
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import pandas as pd
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import sklearn.metrics as sm
from sklearn import datasets
from sklearn.metrics import confusion_matrix,classification_report
# %matplotlib inline
def plotInputs(x):
    fig= plt.figure(figsize=(25,12))
    x1, y1 = map(list, zip(*x))
    plt.scatter(x1,y1,s=300,marker='o', color='b')
    plt.scatter([23],[28],marker='<',color='r')
    plt.scatter([-8],[-5.5],marker='<',color = 'r')
    plt.scatter([21],[27],marker='o',color='r')
    plt.scatter([19],[28],marker='o',color='r')
    plt.scatter([-11],[-6], marker = 'o', color='g')
    plt.scatter([-12],[-8], marker = 'o', color='g')

def demoCluster(x):
    Kmean = KMeans(n_clusters=3)
    Kmean.fit(x)
    print(Kmean.cluster_centers_)
    print(Kmean.labels_)

plotInputs(inputs)

"""## Problem 2: Clustering the Locations
Imagine first we have enough budget for three meetingups. You need to plot your clusters for the three meetingups using different colors. Based on the result, what locations would you choose for the meetups?

Later, assume that we are informed we only have enough budget for two meetups.  You need to create two clusters and plot them again using different colors.  Based on the result, what locations would you choose for the meetups?

"""

def plotInputs2(x):
    x1, y1 = map(list, zip(*x))
    plt.scatter(x1,y1,s=250,marker='o', color='b')
    plt.plot([23],[28],marker='<',color='r')
    plt.plot([-8],[-5.5],marker='<',color = 'r')
    plt.plot([21],[27],marker='o',color='r')
    plt.plot([19],[28],marker='o',color='r')
    plt.plot([-14],[-5], marker = 'o', color='r')
    plt.plot([-11],[-6], marker = 'o', color='r')
    plt.plot([-12],[-8], marker = 'o', color='r')

def plotInputs3(x):
    x1, y1 = map(list, zip(*x))
    plt.scatter(x1,y1,s=500,marker='o', color='b')
    plt.plot([23],[28],marker='<',color='r')
    plt.plot([-8],[-5.5],marker='<',color = 'r')
    plt.plot([21],[27],marker='o',color='r')
    plt.plot([19],[28],marker='o',color='r')
    plt.plot([-14],[-5], marker = 'o', color='r')
    plt.plot([-11],[-6], marker = 'o', color='r')
    plt.plot([-12],[-8], marker = 'o', color='r')
def plotInputs4(x):
    x1, y1 = map(list, zip(*x))
    plt.scatter(x1,y1,s=750,marker='o', color='b')
    plt.plot([23],[28],marker='<',color='r')
    plt.plot([-8],[-5.5],marker='<',color = 'r')
    plt.plot([21],[27],marker='o',color='r')
    plt.plot([19],[28],marker='o',color='r')
    plt.plot([-14],[-5], marker = 'o', color='g')
    plt.plot([-11],[-6], marker = 'o', color='g')
    plt.plot([-12],[-8], marker = 'o', color='g')

def plotInputs5(x):
    fig= plt.figure(figsize=(25,12))
    x1, y1 = map(list, zip(*x))
    plt.scatter(x1,y1,s=1350,marker='o', color='b')
    plt.plot([23],[28],marker='<',color='r')
    plt.plot([-8],[-5.5],marker='<',color = 'r')
    plt.plot([21],[27],marker='o',color='r')
    plt.plot([19],[28],marker='o',color='r')
    plt.plot([-14],[-5], marker = 'o', color='r')
    plt.plot([-11],[-6], marker = 'o', color='r')
    plt.plot([-12],[-8], marker = 'o', color='r')
    plt.plot([15],[15],marker='<',color = 'r')
    plt.plot([11],[15], marker = 'o', color='r')
    plt.plot([13],[13], marker = 'o', color='r')


    ## by increasing the marker size we can visualize where the closest two points are when the markers converge.
    #as the marker size increases we start to see a clustering visually or atleast points that are candidates
    # to watch this change, change the marker size from 100 - 1000


    # we can verify mathematically now

plotInputs5(inputs)

def calDistances(x): ##Euclidian
    fig= plt.figure(figsize=(25,12))
    x1, y1 = map(list, zip(*x))
    nn =0
    nc =1
    distance= 0
    dl1 = []
    na =0
    nb = 1
    dl2 = []
    dl3 = []
    for i in x1: ##manhattan set to 1; I did set the power to 2 for this forumla but the results are not what I expected and so I implented the function below using a distance formula sqrt.((x2-x1)**2 + (y2-y1)**2)
        distance=minkowskiDist(x1[0:1:],x1[na:nb:],p=1)+ minkowskiDist(y1[0:1:],y1[na:nb:],p=1)
        dl2.append(distance)
        na +=1
        nb += 1
    dl2.pop(0)
    print("manhattan distances" ,dl2)
    for i in range(len(x1)):
        x2 = np.array(x1)
        y2 = np.array(y1)
        distance=math.sqrt((x2[0:1]- x2[nn:nc])**2 + (y2[0:1] - y2[nn:nc])**2) #index slicing into the list
        dl1.append(distance)
        nn += 1
        nc +=1
    dl1.pop(0)
    print("Euclidean distances" , dl1)
    plt.plot(dl1)
    plt.plot(dl2)
    plt.plot([16],[3.1622776601683795], 'ro')
    plt.plot([16],[4.0], 'ro')
    dl1.sort()
    print(dl1[0:1:])
    print(x1,y1)

def calDistances2(x): ##Euclidian
    fig= plt.figure(figsize=(25,12))
    x1, y1 = map(list, zip(*x))
    nn =0
    nc =1
    distance= 0
    dl1 = []
    na =0
    nb = 1
    dl2 = []
    dl3 = []
    for i in x1: ##manhattan set to 1; I did set the power to 2 for this forumla but the results are not what I expected and so I implented the function below using a distance formula sqrt.((x2-x1)**2 + (y2-y1)**2)
        distance=minkowskiDist(x1[4:5:],x1[na:nb:],p=1)+ minkowskiDist(y1[4:5:],y1[na:nb:],p=1)
        dl2.append(distance)
        na +=1
        nb += 1
    dl2.pop(4)
    print("manhattan distances" ,dl2)
    for i in range(len(x1)):
        x2 = np.array(x1)
        y2 = np.array(y1)
        distance=math.sqrt((x2[4:5:]- x2[nn:nc])**2 + (y2[4:5:] - y2[nn:nc])**2) #index slicing into the list
        dl1.append(distance)
        nn += 1
        nc +=1
    dl1.pop(4)
    print("Euclidean distances" , dl1)
    plt.plot(dl1)
    plt.plot(dl2)
    plt.plot([14],[5], 'ro')
    ## visualization of where the closest point on the graph from [-    14,-5](actual points are[-11,-6]), from a Euclidian perspective,
    #plotting my distances against each other for visual comparison. Euclidian is the lined graph starting at 32
    dl1.sort()
    print(dl1[0:1:])
    print(x1)
    print(y1)

calDistances(inputs)

calDistances2(inputs)

def plotCentroid(x):
    fig= plt.figure(figsize=(25,12))
    plt.scatter([-15.88888889], [-10.33333333],s=500, marker= 'X')
    plt.scatter([18.33333333 ], [19.83333333],s=500, marker= 'X')
    plt.scatter([-43.8], [5.4],s=500, marker= 'X')
    plt.scatter([-14],[-5],s=300, marker = 'o', color='b')
    plt.scatter([13],[13],s=300, marker = 'o', color='r')
    plt.scatter([20],[23],s=300, marker = 'o', color='r')
    plt.scatter([-19],[11],s=300, marker = 'o', color='b')
    plt.scatter([-9],[-16],s=300, marker = 'o', color='b')
    plt.scatter([21],[27],s=300, marker = 'o', color='r')
    plt.scatter([-49],[15],s=300, marker = 'o', color='g')
    plt.scatter([26],[13],s=300, marker = 'o', color='r')
    plt.scatter([-46],[5],s=300, marker = 'o', color='g')
    plt.scatter([-34],[-1],s=300, marker = 'o', color='g')
    plt.scatter([11],[15],s=300, marker = 'o', color='r')
    plt.scatter([-49],[0],s=300, marker = 'o', color='g')
    plt.scatter([-22],[-16],s=300, marker = 'o', color='b')
    plt.scatter([19],[28],s=300, marker = 'o', color='r')
    plt.scatter([-12],[-8],s=300, marker = 'o', color='b')
    plt.scatter([-41],[8],s=300, marker = 'o', color='g')

demoCluster(inputs)
plotCentroid(inputs)

"""## Problem 3: Writeup

Answer the following question to evaluate the two results (3-clusters vs. 2-clusters).  

* How do you compare the two results? Which one is better?  And why?

"""

## My first intuition when choosing how to group or cluster from a hierarchal perspective was to choose clusters that
## were near each other or were converging. Further down I took their Euclidian distance using two different points
## calculated their distances between all other points on the graph and out of curiousity I compared using a linear graph
## And thats why you see the graph above here.
#We can see that linearly theyre[manhattan and euclidean] identical in--
# their pattern, not values and this is the calculation for points [-14,-5]
# We see that on this graph the point closest is the 16th value in the list --
## and that the 16th value in our list is the shortest distance
#I have marked the point on the graph with a red dot.
# meaning points [11,-6] is closest to [-14,-5]


# The distance is  Euclidian [3.1622776601683795] Manhattan = 4
## however this is a tedious process and to save time I implemented a Kmeans algorithm too as you can see
## in the above grap where points [-15.88888889 ,-10.33333333],[ 18.33333333 ,19.83333333], [-43.8  ,5.4 ]]
## therefore, the X 's' in the graph above can be the central meeting spot, hypothetically speaking it would be a meeting spot.

## Comparing the two, I personally like using a K-means algorithm, it is much faster than updating a recalculating.

"""# Group Customers
In the following three problems, you need to group customers based on their shopping features.   The data file (shoppingdata.csv) of this part of the assignment can be downloaded from D2L site with this assignment specification. You should explore the data in the file first before your approach the below problems. The dataset has five columns including Annual Income and Spending Score. In this assignment, you are quired to retrieve the last two of these five columns. You need to make use of the Annual Income (in thousands of dollars) and Spending Score (1-100) columns to build your data examples. The Spending Score column signifies how often a person spends money in a mall on a scale of 1 to 100 with 100 being the highest spender.

## Problem 4: Plotting the Customers
For this problem, you need to plot the customers so that their spending scores and annual incomes could be visualized easily.
"""

cd desktop

def getData1(fileName):
    dataFile = open(fileName, 'r')
    cid = []
    genre = []
    age = []
    AI = []
    spc= []
    dataFile.readline()
    for line in dataFile:
        i,g,a,c,s = line.split(',')
        cid.append(int(i))
        genre.append(str(g))
        age.append(int(a))
        AI.append(int(c))
        spc.append(int(s))
    dataFile.close()
    return (AI,spc)
def getData2(fileName):
    dataFile = open(fileName, 'r')
    cid = []
    genre = []
    age = []
    AI = []
    spc= []

    dataFile.readline()
    for line in dataFile:
        i,g,a,c = line.split(',')
        cid.append(int(i))
        genre.append(str(g))
        age.append(int(a))
        AI.append(int(c))
    dataFile.close()
    return (AI)

def plotCustomers(x):
    fig= plt.figure(figsize=(25,12))
    xVals,yVals = getData1(x)
    plt.scatter(xVals, yVals, s=100)

plotCustomers('shoppingdata.csv')

"""## Problem 5: Clustering the Customers

Cluster the customers into different groups using k-means cluster algorithm.  You need to decide what k value you would like to use in your final clustering result.  


"""

def demoCluster2(x):
    fig= plt.figure(figsize=(25,12))
    xVals,yVals = getData1(x)
    XY = [i for i in zip(xVals, yVals)]
    Kmean = KMeans(n_clusters=3)
    Kmean.fit(XY)
    colors = np.array([x for x in 'bgryc'])
    colors = np.hstack([colors] * 20)
    centers = Kmean.labels_
    center_colors = colors[:len(centers)]
    plt.scatter(xVals, yVals, s = 300, c=Kmean.labels_.astype(float))
    plt.scatter([44.15447154], [49.82926829], s = 800, marker = 'X')
    plt.scatter([86.53846154], [82.12820513], s = 800, marker = 'X')
    plt.scatter([87], [18.63157895], s = 800, marker = 'X')

demoCluster2('shoppingdata.csv')

"""## Problem 6: Writeup
How did you choose your k value in your final result?  And how do you evaluate your final result?
.
"""

## For my K value I set it to 3; however, I did jump around to various values for k like 4 and 5 and
## from a visual perspective I found various points improperly clustered.
## For example I just found point closer to other clusters than the cluster that they were labeled as
## Which to me was verification that it was not properly labeling or clustering the data properly

"""# Turn-in

Turn in your notebook including your Python code and answers to the questions to D2L Assignments folder <b>Assignment 7</b>. In addition to the notebook document, you need to provide a pdf that has execution output from each code cell in your notebook.
"""