import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
a = np.arange(24)
print(a.ndim)  # a 现只有一个维度
print(a)
# 现在调整其大小
b = a.reshape(2, 4, 3)  # b 现在拥有三个维度
print(b.ndim)
print(b)

x = np.zeros(5)
print(x, '\n')

c = np.linspace(1, 10, num=50, endpoint=True, retstep=True, dtype=None)
print(c)

x = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
print('我们的数组是：')
print(x)
print('\n')
rows = np.array([[0, 0], [3, 3]])
cols = np.array([[0, 2], [0, 2]])
y = x[rows, cols]
print('这个数组的四个角元素是：')
print(y)

e = np.array([[0, 0, 0],
              [10, 10, 10],
              [20, 20, 20],
              [30, 30, 30]])
f = np.array([1, 2, 3])
ff = np.tile(f, (4, 1))  # 重复 b 的各个维度
print(e + ff)


a = np.arange(6).reshape(2, 3)
print('原始数组是：')
print(a)
print('\n')
print('迭代输出元素：')
for x in np.nditer(a):
    print(x, end=", ")
print('\n')


# 计算正弦曲线上点的 x 和 y 坐标
x = np.arange(0,  3  * np.pi,  0.1)
y = np.sin(x)
plt.title("sine wave form")
# 使用 matplotlib 来绘制点
plt.plot(x, y,'ob')
plt.show()

x =  [5,8,10]
y =  [12,16,6]
x2 =  [6,9,11]
y2 =  [6,15,7]
plt.bar(x, y, align =  'center')
plt.bar(x2, y2, color = 'y' , align =  'center')
plt.title('Bar graph')
plt.ylabel('Y axis')
plt.xlabel('X axis')
plt.show()


mydataset = {
  'sites': ["Google", "Runoob", "Wiki"],
  'number': [1, 2, 3]
}

myvar = pd.DataFrame(mydataset)

print(myvar,'\n')

a = ["Google", "Runoob", "Wiki"]

myvar = pd.Series(a, #index = ["x", "y", "z"] )

print(myvar)


df = pd.read_csv('/Users/fanxuancheng/Desktop/nba.csv')

print(df)
"""

df = pd.read_csv('/Users/fanxuancheng/Desktop/舔列.csv', low_memory=False)
# print(df.info())
# df.drop(df.columns[[0,1,2]], axis=1,inplace=True) #axis=1表示对列操作，删除指定索引列
# df.drop(df.index[200:254364], inplace=True) #删除指定索引行  留下二十多行测试代码
# df.insert(1, '基金状态', '')
# df.insert(2, '相关系数（皮尔逊相关）', '') #在第一列后面添两列
df1 = df[df["c_fund_full_name"].str.contains("可转", na=True)]
df2 = df[df["c_fund_name_x"].str.contains("可交", na=True)]
df3 = df[df["c_fund_name_x"].str.contains("转债", n a=True)]
print(df.head())
#怎么把三个表格合并起来
dfnew = merge(df1, df2, on='序列号')
dfnew.to_csv('/Users/fanxuancheng/Desktop/筛选.csv')

print('123')

print('git 3')
print('git 4')
