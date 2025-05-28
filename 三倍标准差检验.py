"""
仅限正态分布
不适合小样本
极端值会对结果产生影响
"""

import numpy as np
import pandas as pd


def remove_outliers_3sigma(data):
    data = np.array(data)
    # 计算均值和标准差
    mean_val = np.mean(data)
    std_dev = np.std(data)

    # 定义三倍标准差范围
    lower_bound = mean_val - 3 * std_dev
    upper_bound = mean_val + 3 * std_dev

    # 筛选出在这个范围内的数据点

    filtered_data = data[(data >= lower_bound) and (data <= upper_bound)]

    return filtered_data

if __name__ == '__main__':
    data=pd.read_excel('预处理数据.xlsx',sheet_name='Sheet2')
    data=data['销量']

    print(len(data))
    a=remove_outliers_3sigma(data)
    print(len(a))