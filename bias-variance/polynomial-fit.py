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

c_true = [0.3, -1.5, 1.2, 1.5, 0]

x = np.linspace(0, 3.5, 100, endpoint=False)
y = np.polyval(c_true, x)

x_h = np.linspace(2, 4.5, 50, endpoint=False)
y_h = np.polyval(c_true, x_h)

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

y += np.random.normal(0, 0.7, len(y)) + np.random.normal(1, 0.2, len(y))
y_h += np.random.normal(0, 0.7, len(y_h)) + np.random.normal(1, 0.2, len(y_h))

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
poly = [3, 4, 5, 9]
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
n_boot = 200
# poly = [3, 4, 5, 9]
poly = [3]
np.random.seed(2019)
for p, c in zip(poly, col):
    fig, ax = plt.subplots(1, 2, figsize=(10, 7))
    axes = ax.ravel()
    axes[0].plot(x, y,
                 '.', color='deeppink', markeredgecolor=None, alpha=0.5,
                 label='train')
    axes[1].plot(x_h, y_h,
                 '.', color='darkcyan', markeredgecolor=None, alpha=0.5,
                 label='test')
    y_ave_tr = np.zeros(len(x))
    y_tr_df = pd.DataFrame({'y_train_true': np.polyval(c_true, x)})
    y_ave_te = np.zeros(len(x_h))
    for b in range(n_boot):
        samp = np.random.choice(range(len(x)), int(0.5 * len(x)))
        samp = np.sort(samp)
        cc = np.polyfit(x[samp], y[samp], p)
        y_tr = np.polyval(cc, x[samp])
        y_ave_tr += np.polyval(cc, x) / n_boot
        y_te = np.polyval(cc, x_h)
        y_tr_df['y_train_{}'.format(b)] = np.polyval(cc, x)
        y_ave_te += y_te / n_boot
        axes[0].plot(x[samp], y_tr, '-', color=c, alpha=0.05)
        axes[1].plot(x_h, y_te, '-', color=c, alpha=0.05)

    # loss = ((y_tr_df.iloc[:, 1:].values - y_tr_df[
    #     'y_train_true'].values.reshape(-1, 1)) ** 2 / len(
    #     x) / n_boot).sum().sum()
    loss = ((y_tr_df.iloc[:, 1:].values - y.reshape(-1, 1)) ** 2 / len(
        x) / n_boot).sum().sum()

    bias2 = ((y_ave_tr - y_tr_df['y_train_true']) ** 2 / len(x)).sum()
    variance = ((y_tr_df.iloc[:, 1:].values - y_ave_tr.reshape(-1, 1)) ** 2 /
                len(x) / n_boot).sum().sum()
    noise = ((y_tr_df['y_train_true'].values - y) ** 2 / len(x)).sum()

    eq_str = 'LOSS = BIAS^2 + VARIANCE + NOISE' + \
             '\n{:.3f} = {:.3f} + {:.3f}+ {:.3f}'.format(
                 loss, bias2, variance, noise)
    print(eq_str)

    axes[0].plot(x, y_ave_tr, '-', color=c, alpha=1)
    axes[1].plot(x_h, y_ave_te, '-', color=c, alpha=1)
    axes[0].set_ylim(-3, 3.5)
    axes[1].set_ylim(-1.5, 2.75)
    axes[0].legend()
    axes[1].legend()

    # TODO : bias and variance => train
    axes[0].set_title('mse : {:.3f}\n{}'.format(
        mean_squared_error(y, y_ave_tr), eq_str), fontsize=8)
    axes[1].set_title('mse : {:.3f}'.format(
        mean_squared_error(y_h, y_ave_te)), fontsize=10)
    fig.suptitle('poly {} fit with bootstrap'.format(p),
                 fontsize=15)
    fig.show()
    fig.savefig('./bias-variance/poly{}fit-w-bootstrap.png'.format(p),
                dpi=220)
