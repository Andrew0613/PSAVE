import numpy as np
import matplotlib.pyplot as plt


image_size = 28
width = 1
sp_num = image_size // width
met_path = "./metric/weight.txt"
fig_path = "./figure/weight.png"
res = -np.loadtxt(met_path)

display = np.reshape(res, (sp_num, sp_num))
plt.imshow(display, cmap="seismic")
plt.savefig(fig_path)
plt.show()
