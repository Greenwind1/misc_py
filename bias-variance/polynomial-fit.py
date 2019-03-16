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

SEED = 2019

np.random.seed(SEED)
c_true = [0.3, -1.5, 1.2, 1.5, 0]

x = np.linspace(0, 3.5, 200, endpoint=False)
y = np.polyval(c_true, x)

x_h = np.linspace(2, 4.5, 200, endpoint=False)
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
poly = [3, 4, 6, 9]
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
                 label='poly{} mse : {:.4f}'.format(
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
    rmse_tr += np.dot(coef[:-1], coef[:-1]) / len(x)
    # rmse_tr += np.sum(np.abs(coef[:-1])) / len(x)
    print(np.sum(np.abs(coef[:-1])) / len(x))
    axes[0].plot(x, y_tr, '-', color=c, alpha=0.9,
                 label='poly{} mse+penalty : {:.4f}'.format(
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
n_samp = 5000
n_boot = 500
samp_rate = 0.2
poly = [3, 4, 6, 9]
# poly = [3]
np.random.seed(2019)

x_samp = np.random.uniform(0, 3.5, n_samp)
y_samp = np.polyval(c_true, x_samp)
y_samp += np.random.normal(0, 0.7, n_samp)  # SD = 0.7

for p, c in zip(poly, col):

    noise = 0
    loss = 0

    fig, ax = plt.subplots(1, 2, figsize=(10, 7))
    axes = ax.ravel()
    axes[0].plot(x_samp, y_samp,
                 '.', color='deeppink', markersize=0.5,
                 markeredgecolor=None, alpha=0.5,
                 label='train')
    axes[1].plot(x_h, y_h,
                 '.', color='darkcyan', markeredgecolor=None, alpha=0.5,
                 label='test')
    axes[0].set_ylim(-3, 3.5)
    axes[1].set_ylim(-3, 3.5)

    y_ave_tr = np.zeros(len(x_samp))
    y_tr_df = pd.DataFrame({'y_train_true': np.polyval(c_true, x_samp)})
    y_ave_te = np.zeros(len(x_h))

    for b in range(n_boot):
        samp = np.random.choice(range(len(x_samp)),
                                int(samp_rate * len(x_samp)),
                                replace=False)
        samp = np.sort(samp)
        cc = np.polyfit(x_samp[samp], y_samp[samp], p)
        y_tr = np.polyval(cc, x_samp[samp])
        y_ave_tr += np.polyval(cc, x_samp) / n_boot
        y_te = np.polyval(cc, x_h)
        y_tr_df['y_train_{}'.format(b)] = np.polyval(cc, x_samp)
        y_ave_te += y_te / n_boot
        # axes[0].plot(x_samp[samp], y_tr,
        #              '-', color=c, alpha=0.1, linewidth=0.5)
        axes[1].plot(x_h, y_te, '-', color=c, alpha=0.1, linewidth=0.5)

        noise += ((y_tr_df['y_train_true'].values - y_samp) ** 2).mean()
        loss += ((np.polyval(cc, x_samp) - y_samp) ** 2).mean()

    loss /= n_boot
    noise /= n_boot
    bias2 = ((y_ave_tr - y_tr_df['y_train_true']) ** 2).mean()
    variance = ((y_tr_df.iloc[:, 1:].values - y_ave_tr.reshape(-1, 1)) **
                2).mean().mean()
    eq_str = 'LOSS = {:.4f}\nBIAS^2 + VARIANCE + NOISE = LOSS'.format(loss) + \
             '\n{:.4f} + {:.4f}+ {:.4f} = {:.4f}'.format(
                 bias2, variance, noise, bias2 + variance + noise)
    print(eq_str)  # wired...

    title_sqr = 'BIAS^2 : {:.4f}\nVARIANCE : {:.4f}\nNOISE : {:.4f}'.format(
        bias2, variance, noise)
    axes[0].plot(x_samp, y_ave_tr, '.', color=c,
                 alpha=0.1, markersize=0.3, markeredgecolor=None)
    axes[1].plot(x_h, y_ave_te, '--', color=c, alpha=1, linewidth=3)
    axes[0].legend()
    axes[1].legend()
    axes[0].set_title('{}'.format(title_sqr), fontsize=10)
    axes[1].set_title('mse : {:.3f}'.format(
        mean_squared_error(y_h, y_ave_te)), fontsize=10)
    fig.suptitle('poly {} fit with bootstrap'.format(p), fontsize=15)
    fig.show()
    fig.savefig('./bias-variance/poly{}fit-w-bootstrap.png'.format(p),
                dpi=220)
