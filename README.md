# soundmonitor
Record sound, detect events and send alarms for predefined pattern using machine learning.

This project mainly wrote in python.

##  Record sound on laptop to wav file
Usage: record1.py <filename> <seconds>
Example:
```python record1.py  jing1.wav 30```


##  Split sound into small files by 2 seconds
Example:
```python split_sound.py jing1.wav```

Output files under debug folder.
Files under "debug" folder.

## Generate fingsprint for wav files under "debug" folder
```
python spectrum2xy.py  ==> generate data.X, and data.Y for training
python data_label.py  ==> update label of data in data.Y
```

Check output file: data.X, data.Y

## analysis sound files with XGBoost and TSNE

```
python xgb1.py
python data2class.py plot test
```

## Plot TSNE and play sound file on click event
Jupyter notebook file: onclick

