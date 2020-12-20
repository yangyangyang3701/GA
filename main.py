import numpy as np
import GA
import pandas as pd
import matplotlib.pyplot as plt


# 定义fitness function
def schaffer(p):
    x1, x2 = p
    x = x1 ** 2 + (x2 - 1) ** 2
    return x


# 注意调用class的写法
# func是fitness function，n_dim是变量维度，lb是变量下界，precision是变量的精度
ga = GA.GA(func=schaffer, n_dim=2, size_pop=100, max_iter=50, lb=[100, 100], ub=[1000, 1000], precision=1e-7)
best_x, best_y = ga.run()
print('best_x:', best_x, '\n')
print('best_y:', best_y)  # 对应的fitness

Y_history = pd.DataFrame(ga.all_history_Y)
# y的变化
Y_history.min(axis=1).cummin().plot(kind='line')
plt.show()
