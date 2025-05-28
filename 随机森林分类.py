from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def xunlian(x,y,pre_x):
    """
    :param x: 训练集的x
    :param y: 训练集的y
    :param pre_x: 需要进行预测的x
    :return: 预测结果
    """

    # 划分训练集与测试集，比例为8：2，打乱数据顺序，随机数种子设置为42
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # 创建随机森林模型,设定n_estimatorss树的数量为100，随机数种子设置为42
    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    # 训练模型
    rf.fit(X_train, y_train)

    # 预测测试集
    y_pred = rf.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f'准确率: {accuracy:}')

    #根据输入数据进行预测
    pre_y=rf.predict(pre_x)

    return pre_y