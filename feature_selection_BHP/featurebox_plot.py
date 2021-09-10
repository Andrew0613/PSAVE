import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_excel("feature_important.xlsx", header = None)
mean = pd.read_excel("feature_mean.xlsx", header = None)

data = np.array(data)
mean = np.array(mean)

psave = data[:20]
sage = data[20:]

psave_mean = mean[0]
sage_mean = mean[1]


# plt.style.use('ggplot')
y = [
     [0.8727272727272727, 0.8909090909090909, 0.7272727272727273, 0.8, 0.8, 0.7454545454545455, 0.8909090909090909, 0.8181818181818182, 0.7272727272727273, 0.8181818181818182, 0.8545454545454545, 0.7636363636363637, 0.8727272727272727, 0.7818181818181819, 0.8181818181818182, 0.7636363636363637, 0.6545454545454545, 0.6363636363636364, 0.7090909090909091, 0.9090909090909091],[0.4961071144916911, 0.478099416544477, 0.5161156677663734, 0.5157128238025416, 0.5331153011974584, 0.5087251038611925, 0.5280539772727273, 0.5030661962365591, 0.47630284701857284, 0.5125168010752689, 0.5236570595063539, 0.5679317112658846, 0.5231778470185728, 0.5056340878543499, 0.49891174853372433, 0.49725455156402737, 0.5412790964076246, 0.5089255712365591, 0.5186281922043011, 0.5340909090909091]
     ]

plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

# plt.bar(x = [1,2],height = np.mean(np.array(y),1),label='PSAVE',color='cornflowerblue',width = 0.16)
plt.bar(x = [1,2],height = np.median(np.array(np.array(y)*100),1),label='PSAVE',color='cornflowerblue',width = 0.152)

plt.boxplot(x =np.array(np.array(y)*100).T)
plt.xticks([1,2], ("FC","GC"))
plt.ylim(0,100)
# plt.ylim(-0.000003,0.00002)

plt.tick_params(top='off', right='off')

# plt.xlabel("Features",fontsize = 15)
plt.ylabel("percent",fontsize = 15)

# plt.legend(loc = 'upper right',fontsize = 15)
plt.show()