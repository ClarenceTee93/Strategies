# Load the usual suspects
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import datetime as dt
import yfinance as yf
import random
import os
import warnings
from sklearn.model_selection import TimeSeriesSplit

def fillMissingData(data, method):
    """
    Function to handle missing data, available methods are forward fill (to add on more methods)
    
    Parameters
    ----------
    data :  dataframe

    Returns
    -------
    dataframe
    """
    if method == 'forwardFill':
        data.fillna(method='ffill', inplace=True)

    return data

def computeMomentum(df, numMths, start, end, asset_name):
    """
    Function to compute prior n months return (expand for multiple prior months : eg calc 1 mth and 3 mth together), takes in a datafrmae of price

 
    Parameters
    ----------
    df : dataframe with datetimeindex and a column for Price
    numMths : int
    start : str , in the form YYYY-MM-DD
    end : str , in the form YYYY-MM-DD

 
    Returns
    -------
    df : dataframe
    """
    
    
    df = df.copy()
    df.rename(columns = {df.columns[0]:'Price'}, inplace=True)
    
    sDate = datetime.strptime(start, "%Y-%m-%d")
    eDate = datetime.strptime(end, "%Y-%m-%d")


    allDates = pd.date_range(sDate, eDate, freq='d').to_list()
    all_df = pd.DataFrame(index=allDates, columns=['Price_full'])
    all_df_2 = all_df.join(df.Price).drop(['Price_full'], axis=1).fillna(method='ffill')
    all_df_2.reset_index(inplace=True, names='Date')
    col_lab = 'Mth_Prior_' + str(numMths)
    all_df_2[col_lab] = all_df_2.Date.apply(lambda x: x - pd.DateOffset(months=numMths))


    col_lab_2 = col_lab + '_Price'
    col_lab_3 = col_lab + '_Return'


    full_px_w_prior = all_df_2[['Date', 'Price']].rename(columns = {"Date": col_lab, "Price": col_lab_2})


    full_df = all_df_2.merge(full_px_w_prior, on=col_lab, how='inner')
    full_df[col_lab_3] = (full_df['Price'] / full_df[col_lab_2]) - 1
    full_df.set_index('Date', inplace=True)
    output_df = df.join(full_df.drop(['Price'], axis=1))
    output_df['Asset'] = asset_name

    return output_df.dropna()

def volProxy(data, proxy="GK"):
    """
    Returns a low-frequency proxy for realized volatility on a daily basis.
    Options available are: Garman-Klauss ("GK"), Parkinson ("Park"),  OK ("ok"), Roger-Satchell ("RS") and 
    Garman-Klauss-Yang-Zhang ("GKYZ).

    Parameters
    ----------
    data : dataframe
    proxy : str, default "GK"

    Returns
    -------
    dataframe
    """

    match proxy:
        case "GK":
            const = (2 * np.log(2)) - 1
            openClose = const * np.power(np.log(data.resample('D').CLOSE_PRICE.last() / data.resample('D').OPEN_PRICE.first()), 2)
            highLow = 0.5 * np.power(np.log(data.resample('D').HIGH_PRICE.max() / data.resample('D').LOW_PRICE.min()), 2)
            gk = np.sqrt(highLow - openClose)
            gk.dropna(inplace=True)
            
            return gk
        
        case "Park":
            const = 1 / (4 * np.log(2))
            parkinson = data.resample('D').apply(lambda x: np.sqrt(const * np.power(np.log(x.HIGH_PRICE.max()/x.LOW_PRICE.min()), 2)))
            parkinson.dropna(inplace=True)

            return parkinson
        
        case "ok":
            openClose = abs(np.log(data.resample('D').CLOSE_PRICE.last() / data.resample('D').OPEN_PRICE.first()))
            highLow = np.log(data.resample('D').HIGH_PRICE.max() / data.resample('D').LOW_PRICE.min())
            ok = 0.811 * highLow - 0.369 * openClose
            ok.dropna(inplace=True)

            return ok

        case "RS":
            return None
        
        case "GKYZ":
            return None
        
        case _:
            return "Proxy not available!"

def timeSeriesCV(df, blockSize, numRuns):
    """
    Time series cross validation function, start off with just a basic one outlined in Efron & Tibshirani's text on bootstrap.
    
    Parameters
    ----------
    df : pandas dataframe with a datetime index specified, can take in univariate and multivariate time series
    blockSize : int, to specify the blocksize
    numRuns : int, number of samples to be generated

    Returns
    -------
    dict : key tracks the sample number, value is a list containing train & test dataset.
    """
    nrows = df.shape[0] - blockSize
    dictOut = {}
    for i in range(numRuns):
        idx = random.randint(0, nrows)
        dictOut[i] = [df.iloc[np.r_[0:idx, (idx+blockSize):df.shape[0]]], df.iloc[idx:(idx+blockSize)]]

    return dictOut


def exportPerformanceMetrics(stratRet, baselineRet, freq='D'):
    pd.options.display.max_colwidth = 100
    """
    Export performance metrics of a strategy (or a portfolio of strategies; weights required) against benchmark buy and hold strategy. 
    Takes in a dataframe of daily log returns and DateTime as index, assume there is more than a year's worth of returns.

    Parameters
    ----------
    stratRet : pd.Series with DateTimeIndex
    baselineRet : pd.Series with DateTimeIndex
    freq : Annualization factor (12 for monthly returns, 252 for daily returns)

    Returns
    -------
    dataframe

    """
    # Fed funds rate
    riskFreeRate = 0

    if freq == 'D':
        ann_factor = 252
    elif freq == 'M':
        ann_factor = 12

    def extractMetrics(x):
        nobs = len(x)
        # Annualized Returns
        ret = np.power((1+x).prod(), ann_factor/nobs) - 1
        # Annualized Volatility
        vol = np.sqrt(ann_factor) * np.std(x, ddof=1)
        # Sharpe Ratio
        sharpe = (ret - riskFreeRate) / vol
        # Sortino Ratio
        negRet = x[x < 0]
        vol_negRet = np.std(negRet, ddof = 1) * np.sqrt(ann_factor)
        sortino = ret / vol_negRet
        return ret, vol, sharpe, sortino

    Strat_return, Strat_vol, Strat_sharpe, Strat_sortino = extractMetrics(stratRet)
    Strat_md = drawdownCharts(stratRet)[0]
    Strat_r_to_md = abs(Strat_return / Strat_md)
    Bench_return, Bench_vol, Bench_sharpe, Bench_sortino  = extractMetrics(baselineRet)
    Bench_md  = drawdownCharts(baselineRet)[0]
    Bench_r_to_md = abs(Bench_return / Bench_md)
    
    Output_dict={'Performance Metrics':['CAGR %','Volatility','Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown %','Return/MaxDrawdown'],
            'Strategy':['{:.2f}%'.format(Strat_return*100),'{:.2f}%'.format(Strat_vol*100),'{:.2f}'.format(Strat_sharpe),'{:.2f}'.format(Strat_sortino),'{:.2f}%'.format(Strat_md*100),'{:.2f}'.format(Strat_r_to_md)],
            'Benchmark':['{:.2f}%'.format(Bench_return*100),'{:.2f}%'.format(Bench_vol*100),'{:.2f}'.format(Bench_sharpe),'{:.2f}'.format(Bench_sortino),'{:.2f}%'.format(Bench_md*100),'{:.2f}'.format(Bench_r_to_md)]}

    start = str(stratRet.index[0].date())
    end = str(stratRet.index[-1].date())

    Metrics=pd.DataFrame.from_dict(Output_dict)
    Metrics=Metrics.rename(columns={'Performance Metrics':'Performance Metrics '+'('+start+' till '+end+ ')'})
    Metrics.set_index(Metrics.columns[0], inplace=True)

    return Metrics

def drawdownCharts(assetReturns, benchmark=None, includeBenchmark=None, showPlot=False):
    """
    Function to plot out drawdown evaluation, input has to be a pd.Series of simple returns, noncumulative. 
    Plot is exported if not shown. Option added to include benchmark performance.

    Parameters
    ----------
    assetReturns : pd.Series
    benchmark : pd.Series
    includeBenchmark : bool, default None
    showPlot : bool, default False

    Returns
    -------
    ax

    """
    assetCumRet = (assetReturns + 1).cumprod()
    assetCumRet.dropna(inplace=True)
    highWater = np.maximum.accumulate(assetCumRet)
    dVal = assetCumRet.reset_index().Date.values
    fontSizeAll = 10

    fig, ax = plt.subplots(4,1,figsize=(20, 15), constrained_layout=True, sharex=True)
    fig.suptitle("Performance Evaluation/Drawdown Analysis (" + assetReturns.name + ")")

    # High Water Mark
    ax[0].plot(assetCumRet, label="Returns", color='blue', linewidth=1.5)
    ax[0].plot(highWater, label="High water mark", color='black', linewidth=1.5)
    ax[0].fill_between(dVal, assetCumRet.values, highWater.values, alpha=0.25, color="blue")
    ax[0].set_ylabel("Cummulative Returns (%)", fontsize=fontSizeAll)
    # ax[0].set_xlabel("DateTime", fontsize=fontSizeAll)
    ax[0].legend(loc="best", fontsize=fontSizeAll)
    ax[0].grid(axis='y', linewidth=1.5, alpha=0.5)
    ax[0].xaxis.set_tick_params(which='both', labelbottom=True)

    drawdown = (assetCumRet - highWater) / highWater
    largestDD = drawdown[drawdown == drawdown.min()].reset_index().Date.values
    endPt = drawdown[drawdown == drawdown.min()].values
    
    # Deepest Lake
    ax[1].axhline(0, color="black", linewidth=1.5)
    ax[1].plot(drawdown, label="Drawdown", color="blue")
    ax[1].fill_between(dVal, drawdown.values, np.zeros(np.shape(assetCumRet)[0]), alpha=0.25, color="blue")
    ax[1].plot(np.minimum.accumulate(drawdown), linestyle="dashed", color="red", label="Max drawdown")
    ax[1].vlines(x=largestDD, ymin = endPt, ymax=0, color="red", linewidth=1.7, label="Maximum drawdown")
    ax[1].set_ylabel("Drawdown", fontsize=fontSizeAll)
    # ax[1].set_xlabel("DateTime", fontsize=fontSizeAll)
    ax[1].legend(loc="best", fontsize=fontSizeAll)
    ax[1].grid(axis='y', linewidth=1.5, alpha=0.5)
    ax[1].xaxis.set_tick_params(which='both', labelbottom=True)

    # Widest Lake
    def calc_longest_drawdown(x):
        uwater  = (x - x.cummax()) < 0
        runs    = (~uwater).cumsum()[uwater]
        counts  = runs.value_counts(sort=True).iloc[:1]
        max_dur = counts.iloc[0]
        inds    = runs == counts.index[0]
        inds    = (inds).where(inds)
        start   = inds.first_valid_index()
        end     = inds.last_valid_index()
        return max_dur, start, end

    max_dur, xStart, xEnd = calc_longest_drawdown(assetCumRet)
    startLongestDD_y = assetCumRet.loc[xStart]

    ax[2].plot(assetCumRet, label="Returns", color='blue', linewidth=1.5)
    ax[2].plot(highWater, label="High water mark", color='black', linewidth=1.5)
    ax[2].hlines(y=startLongestDD_y, xmin=xStart, xmax=xEnd, linewidth=3, color="red", label="Longest drawdown")
    ax[2].fill_between(dVal, assetCumRet.values, highWater.values, alpha=0.25, color="blue")
    ax[2].set_ylabel("Drawdown", fontsize=fontSizeAll)
    ax[2].legend(loc="best", fontsize=fontSizeAll)
    ax[2].grid(axis='y', linewidth=1.5, alpha=0.5)
    ax[2].xaxis.set_tick_params(which='both', labelbottom=True)

    if includeBenchmark is not None:
        if benchmark is None:
            raise NameError("Benchmark returns not specified")
        else:
            benchmkCum = (benchmark + 1).cumprod()
            benchmkCum.dropna(inplace=True)
            benchmkCum_highWater = np.maximum.accumulate(benchmkCum)
            benchmkCum_dval = benchmkCum.reset_index().Date.values
            ax[0].plot(benchmkCum, label="Returns (Benchmark)", color='darkorange', linewidth=1.5)
            ax[0].plot(benchmkCum_highWater, label="High water mark (Benchmark)", color='black', linewidth=1.5)
            ax[0].fill_between(benchmkCum_dval, benchmkCum.values, benchmkCum_highWater.values, alpha=0.25, color="darkorange")
            ax[0].legend(loc="best", fontsize=fontSizeAll)

            drawdown_benchmk = (benchmkCum - benchmkCum_highWater) / benchmkCum_highWater
            largestDD_benchmk = drawdown_benchmk[drawdown_benchmk == drawdown_benchmk.min()].reset_index().Date.values
            endPt_benchmk = drawdown_benchmk[drawdown_benchmk == drawdown_benchmk.min()].values
            ax[1].plot(drawdown_benchmk, label="Drawdown (Benchmark)", color="darkorange")
            ax[1].fill_between(benchmkCum_dval, drawdown_benchmk.values, np.zeros(np.shape(benchmkCum)[0]), alpha=0.25, color="darkorange")
            ax[1].plot(np.minimum.accumulate(drawdown_benchmk), linestyle="dotted", color="red", label="Max drawdown (Benchmark)")
            ax[1].vlines(x=largestDD_benchmk, ymin = endPt_benchmk, ymax=0, color="red", linestyle="dotted", linewidth=1.7, label="Maximum drawdown (Benchmark)")
            ax[1].legend(loc="best", fontsize=fontSizeAll)

            max_dur_b, xStart_b, xEnd_b = calc_longest_drawdown(benchmkCum)
            startLongestDD_y_b = benchmkCum.loc[xStart_b]
            ax[2].plot(benchmkCum, label="Returns (Benchmark)", color='darkorange', linewidth=1.5)
            ax[2].plot(benchmkCum_highWater, label="High water mark (Benchmark)", color='black', linewidth=1.5)
            ax[2].hlines(y=startLongestDD_y_b, xmin=xStart_b, xmax=xEnd_b, linewidth=3, color="red", label="Longest drawdown (Benchmark)", linestyle="dotted")
            ax[2].fill_between(benchmkCum_dval, benchmkCum.values, benchmkCum_highWater.values, alpha=0.25, color="darkorange")
            ax[2].legend(loc="best", fontsize=fontSizeAll)

            # Plot relative performance
            bnckmk_nm = benchmark.name
            ax[3].plot(assetCumRet/benchmkCum, label="Relative Performance (Strategy versus " + bnckmk_nm + ")", linewidth=1.5, color="black")
            ax[3].legend(loc="best", fontsize=fontSizeAll)
            ax[3].grid(axis='y', linewidth=1.5, alpha=0.5)
            ax[3].set_xlabel("DateTime", fontsize=fontSizeAll)

    else:
        fig.delaxes(ax[3])
        ax[2].set_xlabel("DateTime", fontsize=fontSizeAll)

    if showPlot:
        plt.show()
        return None
    else:
        plt.close()
        return endPt


