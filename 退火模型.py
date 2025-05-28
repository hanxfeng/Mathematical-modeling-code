"""
模拟退火算法可以理解为给定一个初始温度，和一个初始的解，然后产生一个新解，与旧解比较，更好则替换旧解，更差则按照Metropolis准则的概率决定是否替换旧解
迭代一定次数后(这一过程称为内循环)，降低温度，当温度降至指定数值，结束程序
"""
import math
from random import random

def func(x,y):
    return x**2-x*y*2+y**2  #在这里输入需要求解的方程

class SA():
    def __init__(self,func,iter=100,T0=100,Tf=0.01,alpha=0.99):
        self.func = func    #目标函数
        self.iter = iter    #内循环的迭代次数
        self.T0=T0          #初始温度
        self.T=T0           #当前温度
        self.Tf=Tf          #终止温度
        self.alpha=alpha    #降温系数
        #随机生成一百个x值与y值
        self.x=[random() * 11 - 5 for i in range(iter)]
        self.y=[random() * 11 - 5 for i in range(iter)]
        #创建保存最佳解和温度的变量
        self.most_best = []
        self.history = {'f': [], 'T': []}

        #扰动产生新的解
    def generate_new(self, x, y):
        '''
        不加入限制条件
        x_new=x+self.T*(random()-random())
        y_new=y+self.T*(random()-random())
        '''
        while True:
            x_new=x+self.T*(random()-random())
            y_new=y+self.T*(random()-random())
            if (1<x_new<2) and (1<y_new<2):    #这里假设限制条件是x，y均在-5到5之间
                break
        return x_new,y_new

        # Metropolis准则
    def Metropolis(self,f,f_new):
        if f_new<f:    #最大值改为>
            return 1
        else:
            p=math.exp((f-f_new)/(self.T))#最大值交换f与f_new的位置
            if random()<p:  #最大值交换random（）与p
                return 1
            else:
                return 0

    #获取最优解
    def best(self):
        f_list=[] #用于保存每次迭代的值
        for i in range(self.iter):
            f=self.func(self.x[i],self.y[i])
            f_list.append(f)
        #如果是求最大值，min函数改为max函数
        f_best=min(f_list)

        idx=f_list.index(f_best)
        return f_best,idx  #f_best和idx为在该温度下迭代L次后的最优解和它的下标

    #正式求解
    def run(self):
        count=0
        #外循环，当当前温度小于终止温度时停止
        while self.T>self.Tf:
            #内循环
            for i in range(self.iter):
                f=self.func(self.x[i],self.y[i])        #产生用于Metropolis准则的初始f值
                x_new,y_new=self.generate_new(self.x[i],self.y[i])  #产生新解
                f_new=self.func(x_new,y_new)        #产生新的值
                if self.Metropolis(f,f_new):        #通过Metropolis判断是否接受新的值，如果接受将新值存入数组
                    self.x[i]=x_new
                    self.y[i]=y_new
            #迭代结束后记录该温度下的最优解
            ft,_=self.best()
            self.history['f'].append(ft)
            self.history['T'].append(self.T)
            #将温度按比例下降
            self.T=self.T*self.alpha
            count=count+1

        #获得最优解
        f_best,idx=self.best()
        print(f"F={f_best}, x={self.x[idx]}, y={self.y[idx]}")

if __name__ == '__main__':
    sa=SA(func)
    sa.run()

