import numpy as np
import pandas as pd
import feather


def Log(sr):
    #自然对数函数
    return np.log(sr)

def Rank(sr):
    #列-升序排序并转化成百分比
    return sr.rank(axis=1, method='min', pct=True)

def Delta(sr,period):
    #period日差分
    return sr.diff(period)

def Delay(sr,period):
    #period阶滞后项
    return sr.shift(period)

def Corr(x,y,window):
    #window日滚动相关系数
    return (x.rolling(window).corr(y))

def Cov(x,y,window):
    #window日滚动协方差
    return x.rolling(window).cov(y)

def Sum(sr,window):
    #window日滚动求和
    return sr.rolling(window).sum()

def Prod(sr,window):
    #window日滚动求乘积
    return sr.rolling(window).apply(lambda x: np.prod(x))

def Mean(sr,window):
    #window日滚动求均值
    return sr.rolling(window).mean()

def Std(sr,window):
    #window日滚动求标准差
    return sr.rolling(window).std()

def Tsrank(sr, window):
    #window日序列末尾值的顺位
    return sr.rolling(window).rank(method='min', pct=True)
               
def Tsmax(sr, window):
    #window日滚动求最大值    
    return sr.rolling(window).max()

def Tsmin(sr, window):
    #window日滚动求最小值    
    return sr.rolling(window).min()

def Sign(sr):
    #符号函数
    return np.sign(sr)

def Max(sr1,sr2):
    return np.maximum(sr1, sr2)

def Min(sr1,sr2):
    return np.minimum(sr1, sr2)

def Rowmax(sr):
    return sr.max(axis=1)

def Rowmin(sr):
    return sr.min(axis=1)

def Sma(sr,n,m):
    #sma均值
    return sr.ewm(alpha=m/n, adjust=False).mean()

def Abs(sr):
    #求绝对值
    return sr.abs()

def Returns(df):
    return df.rolling(2).apply(lambda x: x.iloc[-1] / x.iloc[0]) - 1



class Alpha191_first20():

    stock_data = feather.read_dataframe("../data/stk_daily.feather")
    stock_data["date"] = pd.to_datetime(stock_data["date"])
    stock_data["open"] *= stock_data["cumadj"]
    stock_data["high"] *= stock_data["cumadj"]
    stock_data["low"] *= stock_data["cumadj"]
    stock_data["close"] *= stock_data["cumadj"]
    stock_data["vwap"] = stock_data["amount"]/stock_data["volume"]
    stock_data.set_index(["date","stk_id"],inplace=True)

    def __init__(self, stock_list = "default", df_data = stock_data):
        self.stock_list = stock_list
        if self.stock_list != "default":
            df_data = df_data[self.stock_list]
        self.open = df_data['open'] # 开盘价
        self.high = df_data['high'] # 最高价
        self.low = df_data['low'] # 最低价
        self.close = df_data['close'] # 收盘价
        self.volume = df_data['volume'] # 成交量
        self.returns = Returns(df_data['close']) # 日收益率
        self.vwap = df_data['vwap']  # 成交均价
        self.close_prev = df_data['close'].shift(1)#前一天收盘价        
        self.amount = df_data['amount']#交易额

    def alpha001(self): 
        #### (-1 * CORR(RANK(DELTA(LOG(VOLUME), 1)), RANK(((CLOSE - OPEN) / OPEN)), 6))#### 
        return (-1 * Corr(Rank(Delta(Log(self.volume.unstack()), 1)), Rank(((self.close.unstack() - self.open.unstack()) 
                                                                            / self.open.unstack())), 6))
    
    def alpha002(self): 
        #### -1 * delta((((close-low)-(high-close))/(high-low)),1))####
        return -1*Delta((((self.close.unstack()-self.low.unstack())-(self.high.unstack()-self.close.unstack()))
                         /(self.high.unstack()-self.low.unstack())),1) 
    
    def alpha003(self): 
        #### SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6) ####
        cond1 = (self.close.unstack() == Delay(self.close.unstack(),1))
        cond2 = (self.close.unstack() > Delay(self.close.unstack(),1))
        cond3 = (self.close.unstack() < Delay(self.close.unstack(),1))
        part = self.close.unstack().copy(deep=True)
        part[cond1] = 0
        part[cond2] = self.close.unstack() - Min(self.low.unstack(),Delay(self.close.unstack(),1))
        part[cond3] = self.close.unstack() - Max(self.high.unstack(),Delay(self.close.unstack(),1))
        return Sum(part, 6)
    
    def alpha004(self):  
        ####((((SUM(CLOSE, 8) / 8) + STD(CLOSE, 8)) < (SUM(CLOSE, 2) / 2)) ? (-1 * 1) : 
                      # (((SUM(CLOSE, 2) / 2) <((SUM(CLOSE, 8) / 8) - STD(CLOSE, 8))) ? 
                      # 1 : (((1 < (VOLUME / MEAN(VOLUME,20))) || ((VOLUME /MEAN(VOLUME,20)) == 1)) ? 1 : (-1 * 1))))####
        cond1 = ((Sum(self.close.unstack(), 8)/8 + Std(self.close.unstack(), 8)) < Sum(self.close.unstack(), 2)/2)
        cond2 = ((Sum(self.close.unstack(), 8)/8 + Std(self.close.unstack(), 8)) > Sum(self.close.unstack(), 2)/2)
        cond3 = ((Sum(self.close.unstack(), 8)/8 + Std(self.close.unstack(), 8)) == Sum(self.close.unstack(), 2)/2)
        cond4 = (self.volume.unstack()/Mean(self.volume.unstack(), 20) >= 1)
        part = self.close.unstack().copy(deep=True) 
        part[cond1] = -1
        part[cond2] = 1
        part[cond3] = -1
        part[cond3 & cond4] = 1  
        return part
    
    def alpha005(self):
        ####(-1 * TSMAX(CORR(TSRANK(VOLUME, 5), TSRANK(HIGH, 5), 5), 3))###
        return -1*Tsmax(Corr(Tsrank(self.volume.unstack(), 5),Tsrank(self.high.unstack(), 5),5), 3)
    
    def alpha006(self):
        ####(RANK(SIGN(DELTA((((OPEN * 0.85) + (HIGH * 0.15))), 4)))* -1)### 
        return -1*Rank(Sign(Delta(((self.open.unstack() * 0.85) + (self.high.unstack() * 0.15)), 4)))
    
    def alpha007(self):
        ####((RANK(MAX((VWAP - CLOSE), 3)) + RANK(MIN((VWAP - CLOSE), 3))) * RANK(DELTA(VOLUME, 3)))####
        return ((Rank(Tsmax((self.vwap.unstack() - self.close.unstack()), 3)) + 
                 Rank(Tsmin((self.vwap.unstack() - self.close.unstack()), 3))) * Rank(Delta(self.volume.unstack(), 3)))
    
    def alpha008(self): 
        ####RANK(DELTA(((((HIGH + LOW) / 2) * 0.2) + (VWAP * 0.8)), 4) * -1)####    
        return Rank(Delta(((((self.high.unstack() + self.low.unstack()) / 2) * 0.2) + (self.vwap.unstack() * 0.8)), 4) * -1)
    
    def alpha009(self):
        ####SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME,7,2)####  
        return Sma(((self.high.unstack()+self.low.unstack())/2 - 
                    (Delay(self.high.unstack(),1) + Delay(self.low.unstack(),1))/2)*(self.high.unstack()-self.low.unstack())/self.volume.unstack(),7,2)
    
    def alpha010(self):    
        ####(RANK(MAX(((RET < 0) ? STD(RET, 20) : CLOSE)^2),5))####
        cond = (self.returns.unstack() < 0)
        part = self.returns.unstack().copy(deep=True) 
        part[cond] = Std(self.returns.unstack(), 20)
        part[~cond] = self.close.unstack()
        part = part**2
        return Rank(Tsmax(part, 5))
    
    def alpha011(self):
        ####SUM(((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW)*VOLUME,6)####   
        return Sum(((self.close.unstack()-self.low.unstack())-(self.high.unstack()-self.close.unstack()))/(self.high.unstack()-self.low.unstack())*self.volume.unstack(),6)
    
    def alpha012(self):
        ####(RANK((OPEN - (SUM(VWAP, 10) / 10)))) * (-1 * (RANK(ABS((CLOSE - VWAP)))))####   
        return (Rank((self.open.unstack() - (Sum(self.vwap.unstack(), 10) / 10)))) * (-1 * (Rank(Abs((self.close.unstack() - self.vwap.unstack())))))
    
    def alpha013(self): 
        ####(((HIGH * LOW)^0.5) - VWAP)####
        return (((self.high.unstack() * self.low.unstack())**0.5) - self.vwap.unstack())
    
    def alpha014(self): 
        ####CLOSE-DELAY(CLOSE,5)####
        return self.close.unstack()-Delay(self.close.unstack(),5)
    
    def alpha015(self): 
        ####OPEN/DELAY(CLOSE,1)-1####
        return self.open.unstack()/Delay(self.close.unstack(),1)-1
    
    def alpha016(self):   
        ####(-1 * TSMAX(RANK(CORR(RANK(VOLUME), RANK(VWAP), 5)), 5))####
        return (-1 * Tsmax(Rank(Corr(Rank(self.volume.unstack()), Rank(self.vwap.unstack()), 5)), 5))
        
    def alpha017(self):   
        ####RANK((VWAP - MAX(VWAP, 15)))^DELTA(CLOSE, 5)####
        return Rank((self.vwap.unstack() - Tsmax(self.vwap.unstack(), 15)))**Delta(self.close.unstack(), 5)
    
    def alpha018(self):   
        ####CLOSE/DELAY(CLOSE,5)####
        return self.close.unstack()/Delay(self.close.unstack(),5)  
    
    def alpha019(self):  
        ####(CLOSE<DELAY(CLOSE,5)?(CLOSE-DELAY(CLOSE,5))/DELAY(CLOSE,5):(CLOSE=DELAY(CLOSE,5)?0:(CLOSE-DELAY(CLOSE,5))/CLOSE))####
        cond1 = (self.close.unstack() < Delay(self.close.unstack(),5))
        cond2 = (self.close.unstack() == Delay(self.close.unstack(),5))
        cond3 = (self.close.unstack() > Delay(self.close.unstack(),5))
        part = self.close.unstack().copy(deep=True) 
        part[cond1] = (self.close.unstack()-Delay(self.close.unstack(),5))/Delay(self.close.unstack(),5)
        part[cond2] = 0
        part[cond3] = (self.close.unstack()-Delay(self.close.unstack(),5))/self.close.unstack()
        
        return part
       
    def alpha020(self):      
        ####(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100####
        return (self.close.unstack()-Delay(self.close.unstack(),6))/Delay(self.close.unstack(),6)*100