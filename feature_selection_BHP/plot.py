import matplotlib.pyplot as plt
import numpy as np


x = np.array([i for i in range(1, 10)])
feature_selection_sage = np.array([
    184.5925779,
    176.4216251,
    190.7324034,
    176.4689509,
    180.3274817,
    175.8489792,
    197.8499248,
    176.2046741,
    190.9160206

])
feature_selection_psave = np.array([
    171.8119839,
    177.1734211,
    188.3466734,
    196.2662332,
    183.6614251,
    180.6226239,
    192.6866839,
    191.8835518,
    184.6081207
])

plt.figure()
plt.bar(x, feature_selection_sage, label='feature selection of sage', color='cornflowerblue',width = 0.3)
plt.bar(x+0.3, feature_selection_psave, label='feature selection of psave', color='firebrick', width = 0.3)
plt.ylim(ymin = 150, ymax = 200)
plt.xlabel("Features number")
plt.ylabel("Loss")
plt.legend(loc = 'lower right')

plt.savefig("1.png")
