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
# import feather

# stock_data = feather.read_dataframe("../data/stk_daily.feather")
stock_data = pd.read_csv("../data/stock_data.csv",index_col=0)
stock_data["date"] = pd.to_datetime(stock_data["date"])
stock_data["adj_close"] = stock_data["close"]*stock_data["cumadj"]
stock_data["adj_open"] = stock_data["open"]*stock_data["cumadj"]
start_time = "2022-02-01"
end_time = "2022-12-01"


#记录几个常用的市场指数的股票池
hs300_name = pd.read_excel("./new_data/沪深300指数成分股.xlsx")
hs300_list = hs300_name["代码"].values
zz1000_name = pd.read_csv("./new_data/zz1000_data.csv")
zz1000_list = zz1000_name["stk_id"].unique()
all_stock_list = stock_data["stk_id"].unique()



def reverse_strategy(start_time, end_time, backlength, stock_list_name = "all_stock", stock_list = all_stock_list, 
                     stock_data = stock_data, fee = 0):

    #用户可以选择使用一些常用的市场指数股票池
    if stock_list_name == "hs300":
        stock_list = hs300_list
    if stock_list_name == "zz1000":
        stock_list = zz1000_list

    #筛选出时间区间内需要使用的数据
    used_stock_data = stock_data.loc[(stock_data["stk_id"].isin (stock_list)) & (stock_data["date"] >= start_time)
                                          & (stock_data["date"] <= end_time)]
    used_stock_data.index = range(len(used_stock_data))
    # used_stock_data.set_index(["stk_id","date"],inplace=True)

    #计算相邻两天的收益率
    def roll(df, w, **kwargs):
        v = df.values
        d0, d1 = v.shape
        s0,s1 = v.strides
        a = stride(v, (d0 - (w - 1), w, d1), (s0, s0, s1))
        rolled_df = pd.concat({
            row: pd.DataFrame(values, columns=df.columns)
            for row,values in zip(df.index, a)
        })
        return rolled_df.groupby(level=0, **kwargs)
    
    def bar_rate(signal_pre, signal_now, pre_adj_close, adj_open, adj_close, fee = 0):
        if signal_pre == signal_now:
            return signal_now*(adj_close - pre_adj_close)/pre_adj_close
        elif signal_now == 1:
            return (1 - fee)*adj_close/adj_open - 1
        else:
            return adj_open/pre_adj_close*(1 - fee) - 1

    #计算每只股票的持仓信号和净值变化
    def reverse_on_single_stock(single_stock_data):
        single_stock_data["adj_close"] = single_stock_data["adj_close"].ffill()
        single_stock_data["mean_close"] = single_stock_data["adj_close"].rolling(backlength).mean()
        single_stock_data["mean_close"] = single_stock_data["mean_close"].fillna(0)
        single_stock_data["signal"] = np.where(single_stock_data["mean_close"]>single_stock_data["adj_close"],1,0)
        single_stock_data["signal"] = single_stock_data["signal"].shift(1)
        single_stock_data["signal"] = single_stock_data["signal"].fillna(0)
        single_stock_data["daily_ret"] = roll(single_stock_data, 2).apply(lambda x: bar_rate(x["signal"].iloc[0], x["signal"].iloc[1],
                                                                                             x["close"].iloc[0], x["open"].iloc[1],
                                                                                             x["close"].iloc[1],fee = fee))
        single_stock_data["daily_ret"] = single_stock_data["daily_ret"].shift(1)
        single_stock_data["daily_ret"] = single_stock_data["daily_ret"].fillna(0)
        single_stock_data["single_present_value"] = (1 + single_stock_data["daily_ret"]).cumprod()
        return single_stock_data
    
    #计算选定股票池的净值变化
    used_stock_data = used_stock_data.groupby("stk_id").apply(reverse_on_single_stock)
    used_stock_data.set_index(["date","stk_id"],inplace=True)
    used_stock_ret = used_stock_data["single_present_value"]
    reverse_present_value = used_stock_ret.unstack()
    reverse_present_value = reverse_present_value.ffill()
    reverse_present_value = reverse_present_value.fillna(1)
    reverse_present_value["present_value"] = reverse_present_value.mean(axis = 1)
    reverse_present_value["mean_daily_ret"] = reverse_present_value["present_value"].pct_change()
    reverse_present_value["mean_daily_ret"] = reverse_present_value["mean_daily_ret"].fillna(0)
    print(reverse_present_value[["present_value","mean_daily_ret"]])

    #输出净值曲线和信号
    used_stock_signal = used_stock_data["signal"]
    reverse_signal = used_stock_signal.unstack()    
    return [reverse_present_value[["present_value","mean_daily_ret"]], reverse_signal]