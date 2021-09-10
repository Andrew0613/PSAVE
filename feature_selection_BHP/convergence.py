import matplotlib.pyplot as plt


x = [1,2,3,4,5,6]
x_lable = [10,20,50,100,200,500]
y = [
     722.6360125,
     278.3900369,
     116.5343594,
     48.28012409,
     38.20199871,
     14.49748441
     ]

# plt.style.use('ggplot')

plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'

plt.rcParams['axes.unicode_minus'] = False

plt.plot(x, y, label='The variance of PSAVE',color='firebrick')
# plt.bar(x ,y,label='The variance of PSAVE',color='firebrick',width = 0.5)

plt.xticks(x, x_lable)
# plt.boxplot(x = sage)

# plt.ylim(0,85)

plt.tick_params(top='off', right='off')

plt.xlabel("Iterations",fontsize = 15)
plt.ylabel("Variance",fontsize = 15)

plt.legend(loc = 'upper right',fontsize = 15)
plt.show()