#encoding utf-8
# %matplotlib inline
import sys
import time
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

from sklearn import preprocessing
from sklearn.preprocessing import scale
from sklearn import cluster, covariance, manifold

import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

# http://alexanderfabisch.github.io/t-sne-in-scikit-learn.html
from sklearn.manifold import TSNE
# help(TSNE)
from sklearn.cluster import KMeans
import joblib

do_plot = True
label_tps = {}
cluster_tps = {}
cluster_set = {}

def plot_tsne(X,Y):
    global cluster_tps,cluster_set,label_tps
    le = preprocessing.LabelEncoder()
    le.fit(Y.values)
    Y2 =le.transform(Y)

    #print("labels Y:",len(Y2),Y2)
    names = Y.values
    print "names:", names

    modelfile = "tsne.model"
    if os.path.exists(modelfile):
        #cls = joblib.load(modelfile)
        #X_tsne = cls.transform(X)
        #X_tsne = cls.fit_transform(X)
        X_tsne = joblib.load(modelfile)
    else:
        cls = TSNE(learning_rate=200)
        X_tsne = cls.fit_transform(X)
        #joblib.dump(cls, modelfile)
        joblib.dump(X_tsne, modelfile)
    ny = set(Y2)
    #print("ny:",ny)

    modelfile = "kmean.model"
    n_clusters = 6
    if False: # os.path.exists(modelfile):
        kmeans = joblib.load(modelfile)
    else:
        kmeans = KMeans(n_clusters = n_clusters, random_state=0)
        y_pred = kmeans.fit_predict(X_tsne)
        joblib.dump(kmeans, modelfile)

    labels = kmeans.labels_
    n_labels = labels.max()

    ###############################################################################
    # Learn a graphical structure from the correlations
    print "Tsne shape:",X_tsne.shape

    print("n_labels:",n_labels + 1)

    for i in range(n_labels + 1):
        label_index = np.where(labels == i)
        label_name =  names[label_index]

        c_tps = []
        for n in label_name:
            if label_tps[n]=="NEW": continue
            t = int(label_tps[n])
            if t>1: c_tps.append(t)
        print
        cluster_set[i] =  set(label_name) #,label_name
        cluster_tps[i] = sum(c_tps) #/ len(c_tps)

        print 'Cluster %i: ' % i, "tps: %.0f" % cluster_tps[i], "len:",len(label_name)
        print "\t:", cluster_set[i]
        print

    if not do_plot:
        return y_pred

    figure = plt.figure(figsize=(8, 6))
    ax = plt.subplot(111)

    palette = np.array(sns.color_palette("hls", len(ny)))

    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=palette[Y2.astype(np.int)], edgecolor='', s=8)
    #plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=palette[y_pred.astype(np.int)], edgecolor='', s=8)


    #  labels
    txts = []
    for i in ny:
        pos = X_tsne[Y2 == i, :]
        #print "pos:",pos
        # Position of each label.
        xtext, ytext = np.median(X_tsne[Y2 == i, :], axis=0)
        print (i,"==>",le.classes_[i],"\t tps:",label_tps[le.classes_[i]])
        if int(label_tps[le.classes_[i]]) < 1:
            plotmsg = "<*>New-data"
            txt = ax.text(xtext, ytext, plotmsg, color = 'r', fontsize=12)
        else:
            #plotmsg = "(%d)%s" %(i,label_tps[le.classes_[i]])
            plotmsg = "(%d)%s" %(i,le.classes_[i])
            txt = ax.text(xtext, ytext, plotmsg, fontsize=8)
        #plotmsg = "(%d)" %(i)

        #txt.set_path_effects([PathEffects.Stroke(linewidth=3, foreground="w"),PathEffects.Normal()])
        txts.append(txt)

    colors = ['#4EACC5', '#FF9C34', '#4E9A06']
    for k in range(n_clusters):
        col =  colors[k % 3]
        cluster_center = kmeans.cluster_centers_[k]
        #my_members = k_means_labels == k
        #ax.plot(X[my_members, 0], X[my_members, 1], 'w',
        #        markerfacecolor=col, marker='.')
        ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                markeredgecolor='k', markersize=6)

    plt.savefig("tsne.png")
    plt.show()

    return y_pred

def test1():
    X = pd.read_csv("data.X",index_col=0)
    Y = pd.read_csv("data.Y",index_col=0)
    N = len(X.index)

    group1 = Y['tps'].groupby([Y['label']])
    print "group1:",group1
    for name,group in group1:
        label_tps[name] = "%.0f" % group.mean()

    Y = Y['label'] # .values.reshape(N)
    #label_tps = joblib.load("data.tps")

    print "X:",X.shape
    print "Y",Y.shape
    print "tps:", label_tps

    plot_tsne(X,Y)

def test_file():
    global cluster_tps

    X = pd.read_csv("data.X",index_col=0)
    Y = pd.read_csv("data.Y",index_col=0)
    N = len(X.index)

    group1 = Y['tps'].groupby([Y['label']])
    print "group1:",group1
    for name,group in group1:
        label_tps[name] = "%.0f" % group.mean()

    Y = Y['label'] # .values.reshape(N)
    #label_tps = joblib.load("data.tps")

    print "X:",X.shape
    print "Y",Y.shape
    print "tps:", label_tps

    df_new  = pd.read_csv("data.new", delim_whitespace=True)
    NEW_N = len(df_new)

    y_pred = plot_tsne(X,Y)

    y_out = y_pred[-1*NEW_N:]
    y_out = set(y_out)
    new_tps = [cluster_tps[i] for i in y_out]
    print "pred output:", y_out, " pred tps:", new_tps
    pred_label = [cluster_set[i] for i in y_out]
    print "pred type:",pred_label

def main():
    global do_plot
    for opt in sys.argv[1:]:
        print "run %s with opt:" , sys.argv[0]
        if opt == "plot":
            do_plot = True
        if opt =="new":
            test_file()
        elif opt =="test":
            test1()

if __name__ == "__main__":
    print("start on:",time.strftime('Run time: %Y-%m-%d %H:%M:%S'))
    main()
    print("done : ",time.strftime('Run time: %Y-%m-%d %H:%M:%S'))
