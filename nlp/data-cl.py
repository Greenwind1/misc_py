# -*- coding: utf-8 -*-

import gc
import os
import warnings
import psutil
import json
import pickle
import collections as cl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.options.mode.chained_assignment = None
# dir(pd.options.display)
warnings.simplefilter(action='ignore', category=FutureWarning)

plt.style.use('ggplot')

train = pd.read_csv('./nlp/ner_dataset.zip')
# 47959

train['Sentence #'].fillna(method='ffill', inplace=True)
train.head()

train.to_csv('./nlp/ner_dataset.zip', compression='zip', index=False)
