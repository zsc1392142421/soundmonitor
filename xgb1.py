#encoding: utf-8
import numpy as np
import pandas as pd
import random
from sklearn import ensemble, linear_model, metrics
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
#from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
import  xgboost as xgb

def main():
    #加载文件
    X = pd.read_csv("data.X",index_col=0)
    Y = pd.read_csv("data.Y",index_col=0)
    le = preprocessing.LabelEncoder()
    ylabel = Y['label']
    le.fit(ylabel.values)
    print "le:",le.classes_
    y =le.transform(ylabel)
    #print "y:",y

    #分割训练集和测试集，test_size代表测试集所占的比例
    #X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)


    parameters = [{'n_estimators': [100, 200, 500],
                         'max_depth': [3,5,7,9],
                         'min_child_weight':[1,3,6],
                         'learning_rate': [0.5, 1.0],
                         'subsample': [0.75, 0.8, 0.85, 0.9],
                         'colsample_bytree':[0.75,0.8,0.85,0.9]
                         }]


    #model = grid_search.GridSearchCV(xgb.XGBClassifier(), param_grid=parameters,
    #                                 scoring='precision_macro', cv=5)
    model = xgb.XGBClassifier()
    params = {'max_depth': 5, 'colsample_bytree': 0.75,
            'learning_rate': 0.5, 'n_estimators': 100, 'subsample': 0.75, 'min_child_weight': 6}
    #model = xgb.XGBClassifier(**params)

    model.fit(X_train, y_train)
    #print(model.best_params_)
    y_predicted = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_predicted)
    print("Accuracy:%.2f%%" % (accuracy * 100.0))
    print(metrics.classification_report(y_test, y_predicted))
    print(metrics.confusion_matrix(y_test, y_predicted))
"""
{'max_depth': 3, 'colsample_bytree': 0.75, 'learning_rate': 0.5, 'n_estimators': 100, 'subsample': 0.75, 'min_child_weight': 6}
Accuracy:100.00%
             precision    recall  f1-score   support

p78-180onenode12v12       1.00      1.00      1.00         7
p78-180onenode12v6       1.00      1.00      1.00         7
p78-180twonode12v12       1.00      1.00      1.00         5
p78-180twonode12v6       1.00      1.00      1.00         5

avg / total       1.00      1.00      1.00        24
"""
if __name__ == '__main__':
  main()
