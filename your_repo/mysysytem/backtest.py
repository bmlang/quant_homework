# import feather
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






class performance:

    hs300_data = pd.read_excel("./new_data/沪深300指数日数据.xlsx",index_col="日期", parse_dates = True)
    hs300_data = hs300_data["收盘价(元)"]
    zz1000_data = pd.read_excel("./new_data/中证1000指数.xlsx",index_col="trade_date", parse_dates = True)
    zz1000_data = zz1000_data["收盘价(元)"]

    def __init__(self, start_time, end_time, pv, bench = hs300_data, alter_bench = zz1000_data, benchmark_name = "基准表现"):
        self.start_time = start_time
        self.end_time = end_time
        self.pv = pv.loc[(pv.index >= start_time) & (pv.index <= end_time)]
        self.benchmark_name = benchmark_name
        if self.benchmark_name == "中证1000":
            bench = alter_bench
        self.bench = bench.loc[(bench.index >= start_time) & (bench.index <= end_time)]
        # self.pv = self.pv.reset_index(drop = True)
        # self.bench = self.bench.reset_index(drop = True)

    #净值曲线
    def get_pnl_plot(self):
        plt.figure(figsize = (20,10))
        plt.grid()
        plt.plot(self.bench / self.bench.values[0], linewidth = 3, color = 'b', label = self.benchmark_name)
        plt.plot(self.pv / self.pv.iloc[0], linewidth = 3, color = 'r', label = '策略表现')
        plt.legend(fontsize = 40)
        plt.ylabel(r'账户价值 (万元)', fontsize = 20)
        plt.xlabel(r'时间', fontsize = 20)
        plt.title('净值曲线', fontsize= 20)
        plt.legend(fontsize = 20, loc = 'upper left')
        # plt.savefig('./P&L.png', bbox_inches = 'tight' , dpi=150)

    #超额收益
    def get_excess_return(self):
        annual_ret = (self.pv.iloc[-1] / self.pv.iloc[0]) ** (243 / len(self.pv)) - 1
        annual_benchmark = (self.bench.iloc[-1] / self.bench.iloc[0]) ** (243 / len(self.bench)) - 1
        annual_excess = annual_ret - annual_benchmark
        return ((('%.2f') % (100 * annual_excess)) + "%")

    #年化收益
    def get_return(self):
        annual_ret = (self.pv.iloc[-1] / self.pv.iloc[0]) ** (243 / len(self.pv)) - 1
        return ((('%.2f') % (100 * annual_ret)) + "%")

    #年化波动
    def get_vol(self):
        annual_volatility = np.std(self.pv / self.pv.shift(1) - 1) * np.sqrt(243)
        return ((('%.2f') % (100 * annual_volatility)) + "%")

    #夏普比率
    def get_sharpe_ratio(self):
        annual_volatility = np.std(self.pv / self.pv.shift(1) - 1) * np.sqrt(243)
        annual_ret = (self.pv.iloc[-1] / self.pv.iloc[0]) ** (243 / len(self.pv)) - 1
        sr = annual_ret / annual_volatility
        return (('%.2f') % sr)
    
    #最大回撤
    def get_max_drawdown(self):
        max_drawdown = 0
        highest_price = self.pv.values[0]
        for value in self.pv.values:
            highest_price = max(value, highest_price)
            max_drawdown = max(max_drawdown, 1 - value / highest_price)
        return ((('%.2f') % (100 * max_drawdown)) + "%")

    def get_performance(self):
        performance.get_pnl_plot(self),
        return {
            '开始时间': self.start_time,
            '结束时间': self.end_time,
            '超额收益': performance.get_excess_return(self),
            '年化收益': performance.get_return(self),
            '年化波动': performance.get_vol(self),
            '夏普比率': performance.get_sharpe_ratio(self),
            '最大回撤': performance.get_max_drawdown(self)
        }