import pandas as pd
import numpy as np
import datetime as dt
import pylab
import time as t
import datetime
from datetime import datetime
from datetime import timedelta
import talib

t0 =t.time()

symbols =['PAEL','TPL','SING','DCL','POWER','FCCL','DGKC','LUCK',
          'THCCL','PIOC','GWLC','CHCC','MLCF','FLYNG','EPCL',
          'LOTCHEM','SPL','DOL','NRSL','AGL','GGL','ICL','AKZO','ICI',
           'WAHN','BAPL','FFC','EFERT','FFBL','ENGRO','AHCL','FATIMA',
          'EFOODS','QUICE','ASC','TREET','ZIL','FFL','CLOV',
          'BGL','STCL','GGGL','TGL','GHGL','OGDC','POL','PPL','MARI',
          'SSGC','SNGP','HTL','PSO','SHEL','APL','HASCOL','RPL','MERIT',
          'GLAXO','SEARL','FEROZ','HINOON','ABOT','KEL','JPGL','EPQL',
          'HUBC','PKGP','NCPL','LPL','KAPCO','TSPL','ATRL','BYCO','NRL','PRL',
          'DWSM','SML','MZSM','IMSL','SKRS','HWQS','DSFL','TRG','PTC','TELE',
          'WTL','MDTL','AVN','NETSOL','SYS','HUMNL','PAKD',
          'ANL','CRTM','NML','NCL','GATM','CLCPS','GFIL','CHBL',
          'DFSM','KOSM','AMTEX','HIRAT','NCML','CTM','HMIM',
           'CWSM','RAVT','PIBTL','PICT','PNSC','ASL',
          'DSL','ISL','CSAP','MUGHAL','DKL','ASTL','INIL']

start_date = '2017-01-01'
end_date = '2017-06-02'
exp_return = 0.05


def converter(start_date):
    convert=datetime.strptime(start_date, "%Y-%m-%d")
    return convert

def delta_time(converter,n_days):
    new_date = converter + timedelta(days=n_days)
    return new_date


def data(symbol):
    dates=pd.date_range(start_date,end_date) 
    df=pd.DataFrame(index=dates)
    df_temp=pd.read_csv('/home/furqan/Desktop/python_data/{}.csv'.format(str(symbol)),usecols=['Date','Close','Low','Open','High','Volume'],
                            parse_dates=True,index_col='Date',na_values=['nan'])
    df=df.join(df_temp)
    df=df.fillna(method='ffill')
    df=df.fillna(method='bfill')
    return df

def kse100():
    dates = pd.date_range(start_date,end_date)
    kse_100 = pd.DataFrame(index=dates)
    df_temp=pd.read_csv('/home/furqan/Desktop/python_data/KSE100.csv',usecols=['Date','Close','Open','High','Low','Volume'],
                            parse_dates=True,index_col='Date',na_values=['nan'])
    kse_100 = kse_100.join(df_temp)
    kse_100 = kse_100.fillna(method='ffill')
    kse_100 = kse_100.fillna(method='bfill')
    return kse_100
    

def open_price(df):
    open_price = df['Open']
    open_price = open_price.as_matrix()
    float_open=[float(x) for x in open_price]
    np_float_open = np.array(float_open)
    return np_float_open

def high_price(df):
    high_price = df['High']
    high_price = high_price.as_matrix()
    float_high=[float(x) for x in high_price]
    np_float_high = np.array(float_high)
    return np_float_high

def low_price(df):
    low_price = df['Low']
    low_price = low_price.as_matrix()
    float_low=[float(x) for x in low_price]
    np_float_low = np.array(float_low)
    return np_float_low

def close_price(df):
    close = df['Close']
    close = close.as_matrix()
    float_close=[float(x) for x in close]
    np_float_close = np.array(float_close)
    return np_float_close

def volume(df):
    volume = df['Volume']
    volume = volume.as_matrix()
    float_volume=[float(x) for x in volume]
    np_float_volume = np.array(float_volume)
    return np_float_volume


def rsi_val(closeprice,rsi_date):
    rsi = talib.RSI(closeprice, timeperiod=12)
    dates = pd.date_range(start_date, end_date)
    rsi = pd.DataFrame(rsi,index=dates)
    rsi.columns = ['RSI']
    rsi = rsi.ix[rsi_date: ,]
    return rsi

def exp26(closeprice):
    exp_26 = talib.EMA(closeprice, timeperiod=26)
    dates = pd.date_range(start_date, end_date)
    exp_26 = pd.DataFrame(exp_26, index=dates)
    exp_26.columns=['Exp 26']
    return exp_26

def exp12(closeprice):
    exp_12 = talib.EMA(closeprice, timeperiod=12)
    dates = pd.date_range(start_date, end_date)
    exp_12 = pd.DataFrame(exp_12, index=dates)
    exp_12.columns=['Exp 12']
    return exp_12


def mfi_val(highprice,lowprice,closeprice,volume):
    mfi = talib.MFI(highprice,lowprice,closeprice,volume,timeperiod=14)
    dates= pd.date_range(start_date,end_date)
    mfi=pd.DataFrame(mfi,index=dates)
    mfi.columns=['MFI']
    mfi = mfi.ix[mfi_date: ,]
    return mfi

def ppo_val(closeprice):
    ppo = talib.PPO(closeprice, fastperiod=12, slowperiod=26, matype=1)
    dates= pd.date_range(start_date,end_date)
    ppo=pd.DataFrame(ppo,index=dates)
    ppo.columns=['PPO']
    ppo = ppo.ix[ppo_date: ,]
    return ppo

def will_r(highprice,lowprice,closeprice):
    willr = talib.WILLR(highprice,lowprice,closeprice,timeperiod=12)
    dates=pd.date_range(start_date,end_date)
    willr = pd.DataFrame(willr, index=dates)
    willr.columns = ['Williams %R']
    willr = willr.ix[willr_date: , ]
    return willr

def sharperatio(mean_daily_return,std_daily_return):
    sharpe_ratio = (mean_daily_return)/std_daily_return
    return sharpe_ratio

def downside_val(daily_return):
    s_close = daily_return
    #s_close = np.array(s_close)
    loss=np.zeros((s_close.shape[0],s_close.shape[1]))       
    for i in range(0,s_close.shape[0]):
        for j in range(0,s_close.shape[1]):
            if s_close.ix[i,j]<exp_return:
                loss[i,j]=s_close.ix[i,j]
            else:
               loss[i,j]=0
    return loss

def sortinoratio(downside,mean_daily_return):
    dates = pd.date_range(daily_return_date,end_date)
    downside = pd.DataFrame(downside,index=dates)
    downside.columns = ["Downside"]
    downside = downside[downside.Downside != 0]
    std_downside = downside.std()
    std_downside = std_downside[0]
    sortino_ratio = (mean_daily_return)/std_downside
    return sortino_ratio
    

def dailyreturn(df):
    close = df['Close']
    close = np.array(close)
    dates = pd.date_range(start_date,end_date)
    close = pd.DataFrame(close, index=dates)
    daily_return = (close/close.shift(1))-1
    daily_return = daily_return[1:]
    daily_return.columns = ['Daily Return']
    return daily_return

def yearlyreturn(df):
    close = df['Close']
    start_value = close.ix[0,]
    end_value = close.ix[-1,0]
    yearly_return = (end_value/start_value) - 1
    return yearly_return

def comp_mean(data):
    mean = data.mean()
    mean = mean[0]
    return mean

def comp_std(data):
    std = data.std()
    std = std[0]
    return std

def comp_kurtosis(data):
    kurtosis = data.kurtosis()
    kurtosis = kurtosis[0]
    return kurtosis

def comp_skew(data):
    skew = data.skew()
    skew = skew[0]
    return skew
            

stocks = []

#Get KSE 100
kse_100 = kse100()

for symbol in symbols:
    
    #Get data
    df = data(symbol)

    #Convert start date to date time
    new_date = converter(start_date)

    #Daily return date
    daily_return_date = delta_time(new_date,1)

    #RSI date
    rsi_date = delta_time(new_date,11)

    #MFI date
    mfi_date = delta_time(new_date,14)

    #PPO date
    ppo_date = delta_time(new_date,26)

    #William R Dtae
    willr_date = delta_time(new_date,11)

    #Arranging Data
    openprice = open_price(df)
    closeprice = close_price(df)
    highprice = high_price(df)
    lowprice = low_price(df)
    volume_symbol = volume(df)

    #Calculate Daily Return
    daily_return = dailyreturn(df)

    #Mean daily_return
    mean_daily_return = comp_mean(daily_return)
    std_daily_return = comp_std(daily_return)
    skew_daily_return = comp_skew(daily_return)
    kurtosis_daily_return = comp_kurtosis(daily_return)

    #Calculating Sharpe Ratio
    sharpe_ratio = sharperatio(mean_daily_return,std_daily_return)

    #Yearly Return
    yearly_return = yearlyreturn(df)


    #Calculating downside
    downside = downside_val(daily_return)
    #Calculate Sortino Ratio
    sortino_ratio = sortinoratio(downside,mean_daily_return)

    #Calculating RSI
    rsi = rsi_val(closeprice,rsi_date)
    rsi_current = rsi.ix[-1,]
    rsi_current = rsi_current[0]

    #Calculating william's R
    willr = will_r(highprice,lowprice,closeprice)
    willr_current = willr.ix[-1,]
    willr_current = willr_current[0]
    
    #Calculating MFI
    mfi = mfi_val(highprice,lowprice,closeprice,volume_symbol)
    mfi_current = mfi.ix[-1,]
    mfi_current = mfi_current[0]


    #Calculation PPO
    ppo = ppo_val(closeprice)
    ppo_current = ppo.ix[-1,]
    ppo_current = ppo_current[0]

    if mfi_current < 30:
        stocks.append(symbol)
        
print(stocks)
stockssortino=[]
for stock in stocks:
    
    #Get data
    df = data(stock)

    #Arranging Data
    openprice = open_price(df)
    closeprice = close_price(df)
    highprice = high_price(df)
    lowprice = low_price(df)
    volume_symbol = volume(df)

    #Calculate Daily Return
    daily_return = dailyreturn(df)

    #Mean daily_return
    mean_daily_return = comp_mean(daily_return)
    std_daily_return = comp_std(daily_return)
    skew_daily_return = comp_skew(daily_return)
    kurtosis_daily_return = comp_kurtosis(daily_return)

    #Calculating Sharpe Ratio
    sharpe_ratio = sharperatio(mean_daily_return,std_daily_return)

    #Yearly Return
    yearly_return = yearlyreturn(df)


    #Calculating downside
    downside = downside_val(daily_return)
    #Calculate Sortino Ratio
    sortino_ratio = sortinoratio(downside,mean_daily_return)

    if sortino_ratio > 0:
        stockssortino.append(stock)
        print('------------------------------------------------------------')
        print('Mean Daily Return for ',stock, 'is ',mean_daily_return*100,'%')
        print('Yearly return is ',yearly_return*100,'%')
        print('Standard Deviation of daily return is ', std_daily_return)
        print('Skewness of daily return is ', skew_daily_return)
        if skew_daily_return > 0:
            print('POSITIVELY SKEWED')
        else:
            print('NEGATIVELY SKEWED')
        print('Kurtosis daily return is ', kurtosis_daily_return)
        print('------------------------------------------------------------')  
        print('SHARPE ratio of stock is  ', sharpe_ratio)
        print('SORTINO ratio is ', sortino_ratio)
        print('------------------------------------------------------------')

    
print(stockssortino)


    

t1=t.time()
print('Exec time is ',t1-t0)
