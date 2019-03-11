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
from sklearn.metrics import mean_squared_error

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.options.mode.chained_assignment = None
# dir(pd.options.display)
warnings.simplefilter(action='ignore', category=FutureWarning)

plt.style.use('ggplot')

x = np.linspace(0, 3.5, 100, endpoint=False)
y = np.polyval([0.3, -1.5, 1.2, 1.5, 0], x)

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(x, y, '.', color='deeppink')
ax.set_title('y = {}x^4 + {}x^3 + {}x^2 + {}x + {}'.format(
    0.3, -1.5, 1.2, 1.5, 0))
fig.show()
fig.savefig('./bias-variance/poly.png', dpi=220)

y = np.polyval([0.3, -1.5, 1.2, 1.5, 0], x)
y += np.random.normal(0, 0.7, len(y))

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(x, y, '.', color='deeppink')
ax.set_title('y = {}x^4 + {}x^3 + {}x^2 + {}x + N(0, 0.7^2)'.format(
    0.3, -1.5, 1.2, 1.5, 0))
fig.show()
fig.savefig('./bias-variance/poly-w-gauss.png', dpi=220)

# polynomial fitting
c2 = np.polyfit(x, y, 2)
y2 = np.polyval(c2, x)
c4 = np.polyfit(x, y, 4)
y4 = np.polyval(c4, x)
c6 = np.polyfit(x, y, 6)
y6 = np.polyval(c6, x)
c12 = np.polyfit(x, y, 12)
y12 = np.polyval(c12, x)

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(x, y, '.', color='deeppink')
ax.plot(x, y2, '--', color='darkslateblue',
        label='poly{} mse : {:.3f}'.format(len(c2) - 1,
                                           mean_squared_error(y, y2)))
ax.plot(x, y4, '-', color='dimgray',
        label='poly{} mse : {:.3f}'.format(len(c4) - 1,
                                           mean_squared_error(y, y4)))
ax.plot(x, y6, '--', color='darkorange',
        label='poly{} mse : {:.3f}'.format(len(c6) - 1,
                                           mean_squared_error(y, y6)))
ax.plot(x, y12, '--', color='limegreen',
        label='poly{} mse : {:.3f}'.format(len(c12) - 1,
                                           mean_squared_error(y, y12)))
ax.legend()
fig.show()
fig.savefig('./bias-variance/polyfit.png', dpi=220)

# bayesian view
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(x, y, '.', color='deeppink')
ax.plot(x, y2, '-', color='darkslateblue',
        label='poly{} mse+penalty : {:.2f}'.format(
            len(c2) - 1,
            mean_squared_error(y, y2) + np.dot(c2[:-1], c2[:-1]) / len(x)))
ax.plot(x, y4, '-', color='dimgray',
        label='poly{} mse+penalty : {:.2f}'.format(
            len(c4) - 1,
            mean_squared_error(y, y4) + np.dot(c4[:-1], c4[:-1]) / len(x)))
ax.plot(x, y6, '-', color='darkorange',
        label='poly{} mse+penalty : {:.2f}'.format(
            len(c6) - 1,
            mean_squared_error(y, y6) + np.dot(c6[:-1], c6[:-1]) / len(x)))
ax.plot(x, y12, '-', color='limegreen',
        label='poly{} mse+penalty : {:.2f}'.format(
            len(c12) - 1,
            mean_squared_error(y, y12) + np.dot(c12[:-1], c12[:-1]) / len(x)))
ax.legend()
fig.show()
fig.savefig('./bias-variance/polyfit-w-penalty.png', dpi=220)

# committee
bs = 200
poly = [2, 4, 6, 12]
col = ['darkslateblue', 'dimgray', 'darkorange', 'limegreen']
np.random.seed(2019)
for p, c in zip(poly, col):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(x, y, '.', color='deeppink', alpha=0.5)
    y_ave = np.zeros(len(x))
    for _ in range(bs):
        samp = np.random.choice(range(len(x)), int(0.5 * len(x)))
        samp = np.sort(samp)
        cc = np.polyfit(x[samp], y[samp], p)
        yy = np.polyval(cc, x[samp])
        y_ave += np.polyval(cc, x) / bs
        ax.plot(x[samp], yy, '-', color=c, alpha=0.05)
    ax.plot(x, y_ave, '-', color=c, alpha=1)

    # TODO : bias and variance
    ax.set_title('poly {} fit with bootstrap\nmse : {:.3f}'.format(
        p, mean_squared_error(y, y_ave)
    ))
    fig.show()
    fig.savefig('./bias-variance/poly{}fit-w-bootstrap.png'.format(p),
                dpi=220)
