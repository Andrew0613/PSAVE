import matplotlib.pyplot as plt
import matplotlib
import numpy as np


label_list = ['1', '2', '3', '4','5','6',"7","8"]    # 横坐标刻度显示值
sage = [142.837061,
132.7338165,
145.4540549,
136.202972,
135.4865183,
134.3924717,
143.5192461,
136.1850837


]      # 纵坐标值1
psave = [147.9745923,
139.1990877,
131.4269221,
135.273,
130.5549124,
130.5548219,
133.1580001,
144.4278487


]      # 纵坐标值2
x = range(8)
"""
绘制条形图
left:长条形中点横坐标
height:长条形高度
width:长条形宽度，默认值0.8
label:为后面设置legend准备
"""
plt.bar(x, height=sage, width=0.4, alpha=0.8, color='firebrick', label="SAGE")
plt.bar(np.array(x)+0.4, height=psave, width=0.4, color='cornflowerblue', label="PSAVE")
plt.ylim(130, 150)     # y轴取值范围
plt.ylabel("Loss",fontsize = 15)
"""
设置x轴刻度显示值
参数一：中点坐标
参数二：显示值
"""
plt.xticks([index + 0.2 for index in x], label_list)
plt.xlabel("Number of features",fontsize = 15)

plt.legend(fontsize = 15)     # 设置题注
"""
# 编辑文本
for rect in rects1:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height), ha="center", va="bottom")
for rect in rects2:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height), ha="center", va="bottom")
plt.show()
"""