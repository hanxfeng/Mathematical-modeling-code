from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd


def svr(x,y,x1):
    """
    :param x: 训练集的x
    :param y: 训练集的y
    :param x1: 需要预测的x
    :return: 预测结果
    """
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,shuffle=False)

    # 对特征进行标准化处理
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(x_train)
    X_test_scaled = scaler.transform(x_test)
    x1 = scaler.fit_transform(x1)

    # 创建SVM分类器实例，这里使用线性核函数
    svm_clf = SVR(kernel='rbf', C=1e3, gamma=0.1)  # 这里我们使用RBF核函数，并设置C和gamma参数

    # 训练模型
    svm_clf.fit(X_train_scaled, y_train)

    # 在测试集上进行预测
    y_pred = svm_clf.predict(X_test_scaled)

    # 打印分类报告，评估模型性能
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    print(f"R2: {r2}")

    # 进行预测
    y_re = svm_clf.predict(x1)

    return y_re