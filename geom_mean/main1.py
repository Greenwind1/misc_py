# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

plt.style.use('ggplot')

x = np.linspace(0, 1)
y = np.linspace(0, 1)

X, Y = np.meshgrid(x, y)
z = np.sqrt(X * Y)
zz = (X + Y) / 2

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].pcolor(X, Y, z, cmap=cm.Accent)
ax[0].set_title('geom_mean')
ax[1].pcolor(X, Y, zz, cmap=cm.Accent)
ax[1].set_title('simple_mean')
fig.show()
