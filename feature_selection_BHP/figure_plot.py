import numpy as np
import matplotlib.pyplot as plt


feature_num = 10
met_path = "./metric/psave_10.txt"
fig_path = "./figure/psave_10.png"
fsave = np.loadtxt(met_path)
print(fsave)

bar_width = 0.3
x_name = np.array(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'])[0:feature_num]
x_fsave = np.arange(feature_num)

colors = ['', '', '', '', '', '', '', '', '', '', '', '', ''][0:feature_num]
for i in range(feature_num):
    if fsave[i] > 0:
        colors[i] = 'cornflowerblue'
    else:
        colors[i] = 'firebrick'

rec1 = plt.bar(x_fsave, height=fsave, width=bar_width, color=colors, label='fsave')

plt.legend()
plt.xticks(x_fsave, x_name, rotation=30)
plt.ylabel('metric value')
plt.title('figure')

plt.savefig(fig_path)
plt.show()
