from sklearn.ensemble import RandomForestRegressor  # 用于回归问题
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def sui_ji_sen_lin(X, Y, X1):
    """
    :param X:训练集的x
    :param Y:训练集的y
    :param X1:需要进行预测的x
    :return:
    """

    # 划分训练集与测试集，比例为8：2，不打乱数据顺序
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shufle=False)

    # 创建随机森林模型,设定n_estimatorss树的数量为100，随机数种子设置为36
    model = RandomForestRegressor(n_estimators=100, random_state=36)

    # 训练模型
    model.fit(x_train, y_train)

    # 预测测试集
    y_pred = model.predict(x_test)

    # 计算均方误差mse与r2得分
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'mse:{mse}')
    print(f"r2:{r2}")

    #根据输入数据进行预测
    y = model.predict(X1)

    return y
