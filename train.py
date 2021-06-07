# import library
import numpy as np
import pandas as pd
import tensorflow as tf

import autokeras as ak

train = pd.read_csv('./data/train.tsv', sep="\t", index_col=0)
test = pd.read_csv('./data/test.tsv', sep="\t", index_col=0)
submission = pd.read_csv(
    './data/sample_submit.tsv', sep="\t", index_col=0, header=None)

X = train.iloc[:, 2:].values
y = train.iloc[:, 1].values

X_test = test.iloc[:, 1:].values
clf = ak.StructuredDataClassifier(overwrite=True, max_trials=3)
clf.fit(X, y, epochs=10)

pred = clf.predict(X_test)

submission[1] = pred.astype('int8')
submission.to_csv("submit.tsv", header=None, sep='\t')
