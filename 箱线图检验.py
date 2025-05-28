'''
适用于正态分布的数据
不适用于小样本数据
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def xiangxiantu(df):
    plt.boxplot(df)
    plt.title('箱线图')
    plt.ylabel('值')
    plt.show()

    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1

    # 计算异常值的下界和上界
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # 过滤出异常值
    is_outlier = (df < lower_bound) | (df > upper_bound)
    outliers = df[is_outlier]

    # 保留非异常值
    df_cleaned = df[~is_outlier]
    #print("异常值:", outliers)
    #print("清理后的数据:", df_cleaned)
    return df_cleaned

if __name__ == '__main__':
    data=pd.read_excel('预处理数据.xlsx',sheet_name='Sheet2')
    data=data['销量']

    print(len(data))
    a=xiangxiantu(data)
    print(len(a))