# -*- coding: utf-8 -*-

import time
import optuna
from hyperopt import fmin, tpe, hp, Trials

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

plt.style.use('ggplot')

JOB = 4
TRIAL = 200


def hyp_obj(arg):
    return (1 - arg['x']) ** 2 + 100 * (arg['y'] - arg['x'] ** 2) ** 2


def opt_obj(trial):
    x = trial.suggest_uniform('x', -10, 10)
    y = trial.suggest_uniform('y', -10, 10)
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


# hyperopt
trials = Trials()
space = {'x': hp.uniform('x', -10, 10),
         'y': hp.uniform('y', -10, 10)}
st = time.time()
best = fmin(fn=hyp_obj,
            space=space,
            algo=tpe.suggest,
            max_evals=TRIAL,
            trials=trials)
print('hyperopt elapssed time :{:.3f}(s)'.format(time.time() - st),
      end='\n')
hyp_res = [t['result']['loss'] for t in trials.trials]

# optuna
study = optuna.create_study()

st = time.time()
study.optimize(opt_obj, n_trials=TRIAL, n_jobs=JOB)
print('optuna elapssed time :{:.3f}(s)'.format(time.time() - st),
      end='\n')
for i in study.best_params:
    print('{} : {:.5f}'.format(i, study.best_params[i]), end='\n')
opt_res = [t.value for t in study.trials]

# draw figure
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
xmax = 5
xmin = -xmax
ymax = 10
ymin = -ymax
n = 1001
x1, y1 = np.meshgrid(np.arange(n), np.arange(n))
x1 = x1 / (n - 1) * (xmax - xmin) + xmin
y1 = y1 / (n - 1) * (ymax - ymin) + ymin
phi = (1 - x1) ** 2 + 100 * (y1 - x1 ** 2) ** 2
interval = [0.1, 1, 10, 100, 500, 1000, 2000, 4000, 8000, 16000]
ax[0].contour(x1, y1, phi, interval,
              cmap=cm.Paired, linewidths=1)
ax[0].plot(1, 1, 'o', ms=4, color='deeppink')
ax[0].text(1.5, 0.5, 'Optimal\npoint', fontsize=8, color='deeppink')
ax[1].plot(range(TRIAL), pd.DataFrame(hyp_res).cummin(),
           '.-', color='darkslateblue', alpha=0.5,
           label='hyperopt')
ax[1].plot(range(TRIAL), pd.DataFrame(opt_res).cummin(),
           '.-', color='darkorange', alpha=0.5,
           label='optuna')
ax[1].set_yscale('log')
ax[1].set_xlabel('TRIAL')
ax[1].set_ylabel('LOGARITHMIC OBJECTIVE VALUE')
ax[1].legend()
fig.tight_layout()
fig.show()
fig.savefig('optuna_hyperopt_comp2.png', dpi=140)
