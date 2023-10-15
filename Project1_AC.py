#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 19:01:14 2023

AER850 Project 1

The purpose of this project was to develop ML models based on a provided data set.

@author: Akus Chhabra - 500970974
"""

import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import joblib

def metric_evaluation(df_test, pred):
    
    acc = accuracy_score(np.array(df_test['Step']), pred)
    prec = precision_score(np.array(df_test['Step']), pred, average='weighted')
    f1 = f1_score(np.array(df_test['Step']), pred, average='weighted')
    
    return acc, prec, f1

def confusion_matrix_eval(df_test, pred):
    
    cm = confusion_matrix(np.array(df_test['Step']), pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    plt.figure(dpi = 1000)
    disp.plot()
    plt.show()
    
    return

def median_mean(df):
    median = df.median()
    mean = df.sum()/len(df)
    
    return median, mean

#%% Step 1: Data Processing

df_Data = pd.read_csv('Project_1_Data.csv') # Define dataframe to store data from csv file

# Separate test data for testing from sample data for visualization (80% training, 20% testing)

df_test = df_Data.sample(n=int(len(df_Data)*0.20), random_state=1) # Testing Data
df = df_Data.drop(df_test.index) # Training Data
df = df.reset_index()
df.drop('index', axis=1, inplace=True)


#%% Step 2: Data Visualization

# Creates a unique list of step numbers
uni_steps = list(set(df['Step'].tolist()))

# Define temp variable by separating df based on steps
temp = []
for i in range(len(uni_steps)):
    temp.append(df.loc[df['Step'] == uni_steps[i]])

######### Plot of feature and target variables #########

# Plot X vs Step coordinates
plt.figure(dpi = 1000)
for i in range(len(uni_steps)):
    plt.scatter(temp[i]['X'], temp[i]['Step'], linestyle = 'None', marker='.', s=10, label='Step %d' %(uni_steps[i]))
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(which = 'major')
plt.grid(which = 'minor')
plt.xlabel('X')
plt.ylabel('Step')
plt.title('X-coordinate vs. Step Number')
plt.minorticks_on()

# Plot Y vs Step coordinates
plt.figure(dpi = 1000)
for i in range(len(uni_steps)):
    plt.scatter(temp[i]['Y'], temp[i]['Step'], linestyle = 'None', marker='.', s=10, label='Step %d' %(uni_steps[i]))
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(which = 'major')
plt.grid(which = 'minor')
plt.xlabel('Y')
plt.ylabel('Step')
plt.title('Y-coordinate vs. Step Number')
plt.minorticks_on()

# Plot Z vs Step coordinates
plt.figure(dpi = 1000)
for i in range(len(uni_steps)):
    plt.scatter(temp[i]['Z'], temp[i]['Step'], linestyle = 'None', marker='.', s=10, label='Step %d' %(uni_steps[i]))
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(which = 'major')
plt.grid(which = 'minor')
plt.xlabel('Z')
plt.ylabel('Step')
plt.title('Z-coordinate vs. Step Number')
plt.minorticks_on()


######### Statistical Analysis #########

# Median and mean values
median_x, mean_x = median_mean(df['X'])
median_y, mean_y = median_mean(df['Y'])
median_z, mean_z = median_mean(df['Z'])
median_step, mean_step = median_mean(df['Step'])

# Standard deviation
x_sum, y_sum, z_sum = 0, 0, 0
for i in range(len(df['X'])):
    x_sum += (df['X'][i] - mean_x)**2
    y_sum += (df['Y'][i] - mean_y)**2
    z_sum += (df['Z'][i] - mean_z)**2
sx = np.sqrt(x_sum/(len(df['X'])-1))
sy = np.sqrt(y_sum/(len(df['Y'])-1))
sz = np.sqrt(z_sum/(len(df['Z'])-1))


######### Normal Distributions #########
# X Values
plt.figure(dpi=1000)
Xx = np.linspace(mean_x - 3*sx, mean_x + 3*sx, 100)
plt.plot(Xx, stats.norm.pdf(Xx, mean_x, sx))
plt.grid(which = 'major')
plt.grid(which = 'minor')
plt.xlabel('X-coordinate')
plt.ylabel('Probability Density')
plt.title('Normal Distribution for X-coordinates')
plt.minorticks_on()
plt.show()

# Y Values
plt.figure(dpi=1000)
Yx = np.linspace(mean_y - 3*sy, mean_y + 3*sy, 100)
plt.plot(Yx, stats.norm.pdf(Yx, mean_y, sy))
plt.grid(which = 'major')
plt.grid(which = 'minor')
plt.xlabel('Y-coordinate')
plt.ylabel('Probability Density')
plt.title('Normal Distribution for Y-coordinates')
plt.minorticks_on()
plt.show()

# Z Values
plt.figure(dpi=1000)
Zx = np.linspace(mean_z - 3*sz, mean_z + 3*sz, 100)
plt.plot(Zx, stats.norm.pdf(Zx, mean_z, sz))
plt.grid(which = 'major')
plt.grid(which = 'minor')
plt.xlabel('Z-coordinate')
plt.ylabel('Probability Density')
plt.title('Normal Distribution for Z-coordinates')
plt.minorticks_on()
plt.show()


#%% Step 3: Correlation Analysis

# Heatmaps
plt.figure(figsize=(10,5))
sns.set_theme(style = 'white')
corr = df.corr()
heatmap = sns.heatmap(corr, annot=True, cmap='Blues', fmt='.2g')

# Pairplot
plt.figure(figsize=(14,8))
sns.pairplot(df[['X', 'Y', 'Z', 'Step']], kind="reg")

#%% Step 4: Classification Model Development/Engineering 

X = np.array([df['X'], df['Y'], df['Z']])
X = X.transpose()
Y = np.array(df['Step'])

Xp = np.array([df_test['X'], df_test['Y'], df_test['Z']])
Xp = Xp.transpose()

# Linear Regression Model
line_reg = LinearRegression().fit(X,Y)
line_pred = line_reg.predict(Xp)
line_pred = np.round(line_pred,0)
line_pred = line_pred.astype(int)

parameters = {'copy_X': [True,False], 'fit_intercept':[True,False], 'n_jobs': [True,False], 'positive':[True,False]}
line_clf = GridSearchCV(line_reg, parameters, refit = True, verbose = 3)
line_clf.fit(X, Y)
sorted(line_clf.cv_results_.keys())
print(line_clf.best_params_)

line_pred = line_reg.predict(Xp)
line_pred = np.round(line_pred,0)
line_pred = line_pred.astype(int)


# Decision Tree Model
dec_tree = tree.DecisionTreeClassifier().fit(X,Y)
tree_pred = dec_tree.predict(Xp)

metric_evaluation(df_test, tree_pred)

parameters = {'max_depth': [1, 10, 100, 1000, 10000], 'min_samples_split': [5, 10, 20, 40, 80, 100], 
              'min_samples_leaf': [1, 10, 15, 20, 25, 30, 100], 'criterion': ['gini', 'entropy'], 
              'max_features': [1,2,3,4,5,10]}
tree_clf = GridSearchCV(dec_tree, parameters, refit = True, verbose = 3)
tree_clf.fit(X, Y)
sorted(tree_clf.cv_results_.keys())
print(tree_clf.best_params_)

dec_tree = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 100, max_features = 2, 
                                      min_samples_leaf = 1, min_samples_split = 20).fit(X,Y)
tree_pred = dec_tree.predict(Xp)
metric_evaluation(df_test, tree_pred)


# Support Vector Machines (SVC)
svc = SVC(kernel='linear')
svc.fit(X,Y)

parameters = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['linear','rbf']}
svc_clf = GridSearchCV(SVC(), parameters, refit = True, verbose = 3)
svc_clf.fit(X, Y)
sorted(svc_clf.cv_results_.keys())
print(svc_clf.best_params_)

# Post-tuning
svc = SVC(C=1000, gamma=0.01, kernel='rbf')
svc.fit(X,Y)
SVC_pred = svc.predict(Xp) # Predicted results


#%% Step 5: Model Performance Analysis

######### Metrics Evaluation #########
metric_evaluation(df_test, line_pred) # Linear Regression Model
metric_evaluation(df_test, tree_pred) # Decision Tree Model
metric_evaluation(df_test, SVC_pred) # Support Vector Machines Model

######### Confusion Matrix #########
confusion_matrix_eval(df_test, SVC_pred) # Support Vector Machines Model

#%% Step 6: Model Evaluation

# Save SVC model using joblib
filename = 'final_model.sav'
joblib.dump(svc, filename)

# Load model using joblib
load_model = joblib.load(filename)
data = [[9.375,3.0625,1.51], [6.995,5.125,0.3875], [0,3.0625,1.93], [9.4,3,1.8], [9.4,3,1.3]]
result = load_model.predict(data)
print('Predictions for Model Evaluation: %s' %str(result))
