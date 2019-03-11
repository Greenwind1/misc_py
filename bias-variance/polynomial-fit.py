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

x_h = np.linspace(2, 4.5, 50, endpoint=False)
y_h = np.polyval([0.3, -1.5, 1.2, 1.5, 0], x_h)

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(x, y, '.', color='deeppink', markeredgecolor=None,
        label='train')
ax.plot(x_h, y_h, '.', color='darkcyan', markeredgecolor=None,
        label='test')
ax.set_title('y = {}x^4 + {}x^3 + {}x^2 + {}x + {}'.format(
    0.3, -1.5, 1.2, 1.5, 0))
ax.legend()
fig.show()
fig.savefig('./bias-variance/poly.png', dpi=220)

y += np.random.normal(0, 0.7, len(y))
y_h += np.random.normal(0, 0.7, len(y_h))

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(x, y, '.', color='deeppink', markeredgecolor=None,
        label='train')
ax.plot(x_h, y_h, '.', color='darkcyan', markeredgecolor=None,
        label='test')
ax.set_title('y = {}x^4 + {}x^3 + {}x^2 + {}x + N(0, 0.7^2)'.format(
    0.3, -1.5, 1.2, 1.5, 0))
ax.legend()
fig.show()
fig.savefig('./bias-variance/poly-w-gauss.png', dpi=220)

# polynomial fitting
poly = [2, 4, 6, 12]
col = ['darkslateblue', 'dimgray', 'darkorange', 'limegreen']

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
axes = ax.ravel()
axes[0].plot(x, y, '.', color='deeppink', markeredgecolor=None, alpha=0.5,
             label='train')
ax[1].plot(x_h, y_h, '.', color='darkcyan', markeredgecolor=None, alpha=0.5,
           label='test')
for p, c in zip(poly, col):
    coef = np.polyfit(x, y, p)
    y_tr = np.polyval(coef, x)
    y_te = np.polyval(coef, x_h)
    axes[0].plot(x, y_tr, '-', color=c, alpha=0.9,
                 label='poly{} mse : {:.2f}'.format(
                     len(coef) - 1, mean_squared_error(y, y_tr)))
    axes[1].plot(x_h, y_te, '-', color=c, alpha=0.9,
                 label='poly{} mse : {:.2f}'.format(
                     len(coef) - 1, mean_squared_error(y_h, y_te)))
axes[1].set_ylim(-1.5, 2.75)
axes[0].legend()
axes[1].legend()
fig.show()
fig.savefig('./bias-variance/polyfit.png', dpi=220)

# bayesian view
poly = [2, 4, 6, 12]
col = ['darkslateblue', 'dimgray', 'darkorange', 'limegreen']

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
axes = ax.ravel()
axes[0].plot(x, y, '.', color='deeppink', markeredgecolor=None, alpha=0.5,
             label='train')
axes[1].plot(x_h, y_h, '.', color='darkcyan', markeredgecolor=None, alpha=0.5,
             label='test')
for p, c in zip(poly, col):
    coef = np.polyfit(x, y, p)
    y_tr = np.polyval(coef, x)
    y_te = np.polyval(coef, x_h)
    rmse_tr = mean_squared_error(y, y_tr)
    rmse_tr += + np.dot(coef[:-1], coef[:-1]) / len(x)
    axes[0].plot(x, y_tr, '-', color=c, alpha=0.9,
                 label='poly{} mse+penalty : {:.2f}'.format(
                     len(coef) - 1, rmse_tr))
    axes[1].plot(x_h, y_te, '-', color=c, alpha=0.9,
                 label='poly{} mse : {:.2f}'.format(
                     len(coef) - 1, mean_squared_error(y_h, y_te)))
axes[1].set_ylim(-1.5, 2.75)
axes[0].legend()
axes[1].legend()
fig.show()
fig.savefig('./bias-variance/polyfit-w-penalty.png', dpi=220)

# committee
bs = 200
poly = [2, 4, 6, 12]
col = ['darkslateblue', 'dimgray', 'darkorange', 'limegreen']
np.random.seed(2019)
for p, c in zip(poly, col):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    axes = ax.ravel()
    axes[0].plot(x, y,
                 '.', color='deeppink', markeredgecolor=None, alpha=0.5,
                 label='train')
    axes[1].plot(x_h, y_h,
                 '.', color='darkcyan', markeredgecolor=None, alpha=0.5,
                 label='test')
    y_ave_tr = np.zeros(len(x))
    y_ave_te = np.zeros(len(x_h))
    for _ in range(bs):
        samp = np.random.choice(range(len(x)), int(0.5 * len(x)))
        samp = np.sort(samp)
        cc = np.polyfit(x[samp], y[samp], p)
        y_tr = np.polyval(cc, x[samp])
        y_ave_tr += np.polyval(cc, x) / bs
        y_te = np.polyval(cc, x_h)
        y_ave_te += y_te / bs
        axes[0].plot(x[samp], y_tr, '-', color=c, alpha=0.05)
        axes[1].plot(x_h, y_te, '-', color=c, alpha=0.05)
    axes[0].plot(x, y_ave_tr, '-', color=c, alpha=1)
    axes[1].plot(x_h, y_ave_te, '-', color=c, alpha=1)
    axes[0].set_ylim(-3, 3.5)
    axes[1].set_ylim(-1.5, 2.75)
    axes[0].legend()
    axes[1].legend()

    # TODO : bias and variance
    axes[0].set_title(
        'poly {} fit with bootstrap\nmse : {:.3f}'.format(
            p, mean_squared_error(y, y_ave_tr)))
    axes[1].set_title(
        'poly {} fit with bootstrap\nmse : {:.3f}'.format(
            p, mean_squared_error(y_h, y_ave_te)))
    fig.show()
    fig.savefig('./bias-variance/poly{}fit-w-bootstrap.png'.format(p),
                dpi=220)
