import numpy as np
import pandas as pd
from pylab import *


def main():
    # Data Structure
    s = pd.Series([i * 2] for i in range(1, 11))
    print(type(s))
    dates = pd.date_range("20170301", periods=8)  # 从20170301开始的8天
    # 第一种定义表的方式：8行5列，索引值（行名）是dates，属性值（列名）是A~E
    df = pd.DataFrame(np.random.randn(8, 5), index=dates, columns=list("ABCDE"))
    print(df)
    # 第二种定义表的方式：
    # df.DataFrame({"A": 1,
    #               "B": pd.Timestamp("20170301"),
    #               "C": pd.Series(1, index=list(range(4)), dtype="float32"),
    #               "D": np.array([3] * 4, dtype="float32"),
    #               "E": pd.Categorical(["police", "student", "teacher", "doctor"])})

    # df.DataFrame({"A": 1, "B": pd.Timestamp("20170301"), "C": pd.Series(1, index=list(range(4)), dtype="float32"), "D": np.array([3] * 4, dtype="float32"),"E": pd.Categorical(["police", "student", "teacher", "doctor"])})

    # print(df)
    print(df.head(3))  # 打印前三行
    print(df.tail(3))  # 打印后三行
    print(df.index)
    print(df.values)  # 打印结果是数组
    print(df.T)  # 索引和属性互换
    # print(df.sort(columns="C"))
    print(df.sort_index(axis=1, ascending=False))  # 对列索引（axis=1-->属性值）进行降序排序
    print(df.describe())  # 打印出count、mean、std、min、max、25%、50%、75%

    # 切片
    print(df["A"])  # A列
    print(df[:3])  # 前三行
    print(df["20170301":"20170304"])
    print(df.loc[dates[0]])
    print(df.loc["20170301":"20170304", ["B", "D"]])
    print(df.at[dates[0], "C"])
    print(df.iloc[1:3, 2:4])
    print(df.iloc[1, 4])
    print(df.iat[1, 4])
    # print(df[df.B > 0][df.A < 0])
    print("df[df > 0]" + str(df[df > 0]))  # 小于0的返回NaN
    print(df[df["E"].isin([1, 2])])

    # 设置
    s1 = pd.Series(list(range(10, 18)), index=pd.date_range("20170301", periods=8))
    df["F"] = s1
    df.at[dates[0], "A"] = 0
    df.loc[:, "D"] = np.array([4] * len(df))

    # 拷贝
    df2 = df.copy()
    df2[df2 > 0] = -df2
    print(df2)

    # 缺失值处理
    df1 = df.reindex(index=dates[:4], columns=list("ABCD") + ["G"])
    df1.loc[dates[0]:dates[1], "G"] = 1
    print('====缺失值处理====')
    print(df1)
    print(df1.dropna())  # 删除缺失值所在行
    print(df1.fillna(value=2))  # 缺失值填充2
    print('<===缺失值处理===>')
    # 统计指标
    print('===统计指标===')
    print(df)
    print(df.mean())  # 每列的平均值
    print(df.var())
    s = pd.Series([1, 2, 3, np.nan, 5, 7, 9, 10], index=dates)
    print("--------s")
    print(s.shift(2))  # 所有值后移两位，前两位补NaN，多的值删除
    print("======s")
    print(s.diff())  # 不填表示一阶，填的数字表示多阶
    # print(s.calue_counts())  # 每个值出现的次数
    print(df.apply(np.cumsum))
    # print(df.apply(lambda x: x, max() - min()))
    print('<===统计指标===>')
    # 表的拼接
    pieces = [df[:3], df[-3:]]
    print(pd.concat(pieces))
    left = pd.DataFrame({"key": ["x", "y"], "value": [1, 2]})
    right = pd.DataFrame({"key": ["x", "z"], "value": [3, 4]})
    print(pd.merge(left, right, on="key", how="left"))  # how="inner":所有缺失值都删掉;"outer"所有缺失值都保留
    df3 = pd.DataFrame({"A": ["a", "b", "c", "b"], "B": list(range(4))})
    print(df3.groupby("A").sum())  # 将A列每种属性值求和

    # Time Series
    t_exam = pd.date_range("20170301", periods=10, freq="S")
    print("---Time Series--------")
    print(t_exam)

    # Graph
    ts = pd.Series(np.random.randn(1000), index=pd.date_range("20170301", periods=1000))
    ts = ts.cumsum()

    ts.plot()
    show()

    # File
    df6 = pd.read_csv("./data/test.csv")
    print(df6)
    df7 = pd.read_excel("./data/test.xlsx", "Sheet1")
    print("Excel", df7)
    df6.to_csv("./data/test2.csv")
    df7.to_excel("./data/test2.xlsx")
    pass


if __name__ == '__main__':
    main()
