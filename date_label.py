#encoding utf-8
# %matplotlib inline
import sys
import time
import os
import numpy as np
import pandas as pd

Y = pd.read_csv("data.Y",index_col=0)
Y.rename(columns={"label": "label2"},inplace=True)
a = Y['label2'].values
a2 = [i[:2] for i in a]
print a2
Y['label']=a2
Y.to_csv('data.Y')

print Y.head()
