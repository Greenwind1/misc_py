# -*- coding: utf-8 -*-

import gc
import os
import cv2
import warnings
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.options.mode.chained_assignment = None
# dir(pd.options.display)
warnings.simplefilter(action='ignore', category=FutureWarning)

plt.style.use('ggplot')

img = cv2.imread('./img/diving-top_2.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(img)
ax[0].grid(False)
ax[1].imshow(
    cv2.resize(img, dsize=(int(img.shape[1] / 2), int(img.shape[0] / 2)))
)
ax[1].grid(False)
fig.tight_layout()
