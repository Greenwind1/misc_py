import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from fastFM import mcmc

plt.style.use('ggplot')

u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('./ml-100k/u.user', sep='|', names=u_cols)

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('./ml-100k/u.data', sep='\t', names=r_cols)
ratings['date'] = pd.to_datetime(ratings['unix_timestamp'],
                                 unit='s')

m_cols = ['movie_id', 'title', 'release_date',
          'video_release_date', 'imdb_url']
movies = pd.read_csv('./ml-100k/u.item', sep='|', names=m_cols,
                     usecols=range(5), encoding='latin1')

movie_rating = pd.merge(movies, ratings, on='movie_id')
lens = pd.merge(movie_rating, users, on='user_id')

lens.groupby('user_id').size().hist(bins=50, color='deeppink')
user_stats = lens.groupby('user_id').agg({'rating': [np.size,
                                                     np.mean],
                                          'date': [np.max]})
user_stats.describe()


def load_data(filename, path='./ml-100k/'):
    data = []
    y = []
    with open(path + filename) as f:
        for line in f:
            (user, movieid, rating, ts) = line.split('\t')
            data.append({'user_id': str(user),
                         'movie_id': str(movieid)})
            y.append(float(rating))

    return data, np.array(y)


dev_data, y_dev = load_data('ua.base')
test_data, y_test = load_data('ua.test')

v = DictVectorizer()
X_dev = v.fit_transform(dev_data)
X_test = v.transform(test_data)
np.std(y_test)

X_train, X_dev_test, y_train, y_dev_test = train_test_split(
    X_dev, y_dev, test_size=0.1, random_state=0)

n_iter = 300
step_size = 1
seed = 123
rank = 32  # Matrix Factorization parameter

fm = mcmc.FMRegression(n_iter=0, rank=rank, random_state=seed)
fm.fit_predict(X_train, y_train, X_dev_test)

rmse_dev_test = []
rmse_test = []
hyper_param = np.zeros((n_iter - 1, 3 + 2 * rank), dtype=np.float64)

for nr, i in enumerate(range(1, n_iter)):
    fm.random_state = i * seed
    y_pred = fm.fit_predict(X_train, y_train, X_dev_test,
                            n_more_iter=step_size)
    rmse_test.append(np.sqrt(mean_squared_error(y_pred,
                                                y_dev_test)))
    hyper_param[nr, :] = fm.hyper_param_

values = np.arange(1, n_iter)
x = values * step_size
burn_in = 5
x = x[burn_in:]

fig, axes = plt.subplots(2, 2, sharex=True, figsize=(8, 4))
axes[0, 0].plot(x, rmse_test[burn_in:], label='dev test rmse',
                color='deeppink')
axes[0, 0].legend()
axes[0, 1].plot(x, hyper_param[burn_in:, 0], label='alpha',
                color='deeppink')
axes[0, 1].legend()
axes[1, 0].plot(x, hyper_param[burn_in:, 1], label='lambda_w',
                color='deeppink')
axes[1, 0].legend()
axes[1, 1].plot(x, hyper_param[burn_in:, 3], label='mu_w',
                color='deeppink')
axes[1, 1].legend()
fig.tight_layout()
fig.show()
fig.savefig('./mcmcFM.png', dpi=200)
