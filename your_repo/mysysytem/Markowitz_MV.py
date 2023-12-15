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
import scipy.optimize as scopt
import feather





stock_data = feather.read_dataframe("../data/stk_daily.feather")
stock_data["adj_close"] = stock_data["close"]*stock_data["cumadj"]
stock_data["adj_open"] = stock_data["open"]*stock_data["cumadj"]



def Mean_Variance_strategy(start_time, end_time, freq, stock_list, stock_data = stock_data, fee = 0):

    #计算资产组合方差
    def objfunvar(W, R, target_ret):
        cov=np.cov(R.T) # var-cov matrix
        port_var = np.dot(np.dot(W,cov),W.T) # portfolio variance
        return np.sqrt(port_var)

    #组合优化求解有效前沿
    def calc_efficient_frontier(returns,sellshort=True):
        result_means = []
        result_stds = []
        result_weights = []
        result_utilities = []
        
        means = returns.mean()
        min_mean, max_mean = means.min(), means.max()
        
        nstocks = returns.columns.size
        if sellshort:
            bounds=None
        else:
            bounds = [(0,1) for i in np.arange(nstocks)]
        for r in np.linspace(min_mean, max_mean, 100):
            weights = np.ones(nstocks)/nstocks        
            constraints = ({'type': 'eq', 
                            'fun': lambda W: np.sum(W) - 1},
                        {'type': 'eq', 
                            'fun': lambda W: np.sum(W*means) -r})
            results = scopt.minimize(objfunvar, weights, (returns, r), 
                                    method='SLSQP', 
                                    constraints = constraints,
                                    bounds = bounds)
            if not results.success: # handle error
                raise Exception(results.message)
            result_means.append(np.round(r,4)) # 4 decimal places
            #std_=np.round(np.std(np.sum(returns*results.x,axis=1)),6)
            std_=objfunvar(results.x,returns,r)
            result_stds.append(std_)        
            result_weights.append(np.round(results.x, 5))
            result_utilities.append(np.round(r-5*np.power(std_,2),5))
        return {'Means': result_means, 
                'Stds': result_stds, 
                'Weights': result_weights,
                'Utilities': result_utilities}

    #数据处理
    close_price_data = stock_data.loc[(stock_data["date"] >= start_time)&(stock_data["date"] <= end_time),
                                       ["stk_id","date","adj_close"]]
    close_price_data = close_price_data.loc[close_price_data["stk_id"].isin (stock_list)]
    close_price_data.set_index(["date","stk_id"],inplace=True)
    close_price_data = close_price_data.unstack()

    open_price_data = stock_data.loc[(stock_data["date"] >= start_time)&(stock_data["date"] <= end_time),
                                     ["stk_id","date","adj_open"]]
    open_price_data = open_price_data.loc[open_price_data["stk_id"].isin (stock_list)]
    open_price_data.set_index(["date","stk_id"],inplace=True)
    open_price_data = open_price_data.unstack()

    return_data = close_price_data.pct_change()
    return_data = return_data.fillna(0)
    return_data = return_data["adj_close",]
    change_position_date = return_data.index[range(freq-1,len(return_data), freq)] 

    #求解最小方差组合的信号
    def generate_signal(return_data, change_position_date):
        signal_value = return_data - return_data
        effective_frontier = calc_efficient_frontier(return_data.loc[return_data.index <= change_position_date[0]], sellshort=False)
        strategy_MV = effective_frontier["Stds"].index(min(effective_frontier["Stds"]))
        signal_value.loc[signal_value.index > change_position_date[0]] = effective_frontier["Weights"][strategy_MV]

        for period in range(1, len(change_position_date), 1):
            effective_frontier = calc_efficient_frontier(return_data.loc[(return_data.index <= change_position_date[period]) & 
                                                                        (return_data.index > change_position_date[period - 1])], sellshort=False)
            strategy_MV = effective_frontier["Stds"].index(min(effective_frontier["Stds"]))
            signal_value.loc[signal_value.index > change_position_date[period]] = effective_frontier["Weights"][strategy_MV]
        return signal_value
    

    #基于信号生成净值
    def calculate_present_value(signal_value, open_price_data, close_price_data, fee=fee):
        daily_ret = [0]
        for num in range(1, len(signal_value)):
            signal_now = signal_value.iloc[num]
            signal_pre = signal_value.iloc[num - 1]
            close_price = close_price_data.iloc[num]
            pre_close_price = close_price_data.iloc[num - 1]
            open_price = open_price_data.iloc[num]
            trade_fraction = abs(signal_now - signal_pre).sum()

            if trade_fraction == 0:   #判断是否为换仓日
                present_value = np.dot(close_price, signal_now.T)
                last_value = np.dot(pre_close_price, signal_now.T)
                ret_period = 0 if (last_value == 0) else (present_value / last_value - 1)
                daily_ret.append(ret_period)
            else:  
                last_change_open_price = open_price_data.iloc[num - freq]
                change_position_time_pre = signal_pre * open_price / last_change_open_price
                if change_position_time_pre.sum() > 0:
                    trade_fraction = np.abs(signal_now - change_position_time_pre/change_position_time_pre.sum()).sum()
                present_value = np.dot(close_price, signal_now.T)
                change_position_after = np.dot(open_price, signal_now.T)
                change_position_before = np.dot(open_price, signal_pre.T)
                last_value = np.dot(pre_close_price, signal_pre.T) 

                #将换仓日的收益率分成3段来计算：前一交易日收盘-换仓日开盘；换仓前-换仓后；换仓后-换仓日收盘
                ret_period_1 = 1 if (last_value == 0) else change_position_before / last_value
                ret_period_2 = 1 - fee * trade_fraction
                ret_period_3 = present_value / change_position_after
                # print(ret_period_1, ret_period_2, ret_period_3, trade_fraction)
                daily_ret.append(ret_period_1 * ret_period_2 * ret_period_3 - 1)    
        calc_pv = signal_value.copy()
        calc_pv["daily_ret"] = daily_ret
        calc_pv["present_value"] = (1 + calc_pv["daily_ret"]).cumprod()  

        return calc_pv["present_value"]  
    
    signal_value = generate_signal(return_data, change_position_date)
    present_value = calculate_present_value(signal_value, open_price_data, close_price_data, fee=0.0003)
    print(present_value)

    return [present_value, signal_value]

