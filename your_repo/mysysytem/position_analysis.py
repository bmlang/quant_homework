import feather
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from ipywidgets import interact, FloatSlider
# import warnings
# warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置  
plt.rcParams['axes.unicode_minus'] = False
# import statsmodels.api as sm
import seaborn as sns
from numpy.lib.stride_tricks import as_strided as stride
import scipy.optimize as scopt


def position_analysis(signal, freq_set = "False", freq = "Q", period_num = 10):    #freq为按月/季度频率统计，period为按交易日数量统计，freq_name为决定用哪种方式统计的参数

    #读取行业分类数据
    df_name = pd.read_excel("./new_data/行业分类.xlsx")
    stock_data = feather.read_dataframe("../data/stk_daily.feather")
    stock_data["id_adj"] = stock_data["stk_id"].apply(lambda x: str(x)[:-3])
    dict1 = dict(zip(df_name["证券代码"],df_name["中证一级行业分类简称"]))
    dict2 = dict(zip(stock_data["stk_id"], stock_data["id_adj"]))


    #得到行业权重变化结果
    signal_weight = signal.T
    signal_weight["stock_name"] = signal_weight.index
    signal_weight["中证一级行业"] = signal_weight["stock_name"].apply(lambda x: dict1[dict2[x]])
    signal_weight.drop("stock_name",axis=1, inplace=True)
    signal_weight = signal_weight.groupby("中证一级行业").sum()
    signal_weight /= signal_weight.sum()
    signal_weight_hangye = signal_weight.T
    #按freq计算行业平均权重变化
    if freq_set == "True":
        signal_weight_hangye = signal_weight_hangye.resample(freq).mean()
    else:
        n = len(signal_weight_hangye)
        period_length = n // period_num
        signal_weight_index = range(n - period_num * period_length +period_length - 1, n, period_length)
        signal_weight_hangye = signal_weight_hangye.rolling(period_length).mean()
        signal_weight_hangye = signal_weight_hangye.iloc[signal_weight_index]
    signal_weight_hangye.index = pd.to_datetime(signal_weight_hangye.index)

    #画出行业权重的折线堆叠图
    plt.figure(figsize=(24,8))
    sns.set(style = "darkgrid")
    sns.set(font='SimHei',font_scale=1.5)
    label = signal_weight_hangye.columns
    plt.stackplot(signal_weight_hangye.index, signal_weight_hangye.T, labels=label)

    handles, labels = plt.gca().get_legend_handles_labels()
    # 反转图例位置
    plt.legend(reversed(handles), reversed(labels), loc='upper right', prop={'size': 20})
    plt.show()

    return signal_weight_hangye
