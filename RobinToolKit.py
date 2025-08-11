import pandas as pd
import numpy as np
import scipy.stats
import matplotlib as plt
from scipy.stats import norm
from scipy.optimize import minimize

def get_ffme_returns():
    """
    Load the Fama-french Dataset for the returns of the Top & bottom Decile bt Mkt Cap
    """
    me_m = pd.read_csv("C:/Users/g/OneDrive/Documents/EDHEC/M2/SEM 1/Introduction to Portfolio Construction/data/Portfolios_Formed_on_ME_monthly_EW.csv", index_col = 0, na_values=-99.99)
    rets = me_m[['Lo 10', 'Hi 10']]
    rets.columns = ['SmallCap', 'LargeCap']
    rets = rets/100
    rets.index = pd.to_datetime(rets.index, format="%Y%m").to_period('M')
    return rets


def get_hfi_returns():
    """
    Load the Hedge Fund Indicies Dataset for the returns
    """
    hfi = pd.read_csv("C:/Users/g/OneDrive/Documents/EDHEC/M2/SEM 1/Introduction to Portfolio Construction/data/edhec-hedgefundindices.csv", index_col = 0, parse_dates = True)
    hfi = hfi/100
    hfi.index = hfi.index.to_period('M')
    return hfi


def get_ind_returns():
    """
    Load the industry returns dataset
    """
    ind = pd.read_csv("data/ind30_m_vw_rets.csv", index_col = 0, parse_dates = True)/100
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind


def get_ind_nfirms():
    """
    Load the industry returns dataset
    """
    ind = pd.read_csv("data/ind30_m_nfirms.csv", index_col = 0, parse_dates = True)
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind


def get_ind_size():
    """
    Load the industry returns dataset
    """
    ind = pd.read_csv("data/ind30_m_size.csv", index_col = 0, parse_dates = True)
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind


def Returns_ann(r, period):
    """
    input return series, periods (12 for monthly, 252 for daily)
    output annualized returns
    """
    n = r.shape[0]
    return (1+r).prod()**(period/n) - 1


def Vola_ann(r, period):
    """
    input return series, periods (12 for monthly, 252 for daily)
    output annualized volatility
    """
    return r.std()*(period**0.5)


def Shrape_ratio(returns, rf, period):
    """
    INPUTS
    Return series/ dataframe
    Annualized Risk free rate in decimal
    no. of period in a year (12 for monthly, 252 for daily)
    """
    period_rf = (1+rf)**(1/period) -1
    excess_returns = returns - period_rf
    ann_excess_returns = get_returns_ann(excess_returns, period)
    return ann_excess_returns/get_vola_ann(returns, period)


def Drawdown (return_series: pd.Series):
    """
    Input Fataframe and returns a dataframe with welthindex, previouspeak and drawdown
    """
    wealth_index = 1000*(return_series+1).cumprod()
    previous_peak = wealth_index.cummax()
    drawdown = (wealth_index - previous_peak)/previous_peak
    return pd.DataFrame({"Wealth": wealth_index,
                        "Previous Peak" : previous_peak,
                        "Drawdowns": drawdown})


def Skewness(r):
    demeaned_rets = r - r.mean()
    return (demeaned_rets**3).mean()/(r.std(ddof = 0)**3)


def Kurtosis(r):
    """
    Gives the Expected Kurtosis. Subract 3 from the returned values to get excess Kurtosis
    """
    demeaned_r = r - r.mean()
    return (demeaned_r**4).mean()/(r.std(ddof = 0)**4)


def Semideviation (r):
    demeaned_r = r - r.mean()
    excess_negative = demeaned_r[demeaned_r<0]
    len_excess = (demeaned_r<0).sum()
    return (((excess_negative**2).sum())/len_excess)**0.5


def VaR_historic(r, level= 5):
    """
    Returns Historic Var of given returns at a specified quantile. Default level = 5%
    """
    if isinstance (r,pd.DataFrame):
        return r.aggregate(VaR_historic, level=level)
    elif isinstance (r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("Input must be Pandas Data frame or Series")
        

def VaR_gaussian (r, level=5, modified=False):
    """
    returns gausian var if modified = False (Default). If modified = True then returns Cornish-Fischer VaR
    """
    z = norm.ppf(level/100)
    if modified:
        s = Skewness(r)
        k= Kurtosis(r)
        z = (z + (z**2-1)*s/6
            + (z**3 - 3*z)*(k-3)/24
            - (2*z**3 - 5*z)*(s**2)/36)
    
    return -(r.mean() + z*r.std(ddof=0))


def cVaR_historic(r, level=5):
    if isinstance(r, pd.Series):
        shortfall = r <= -VaRhistoric(r, level=level)
        return -r[shortfall].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cVaR_historic, level=level)
    else:
        raise TypeError("Input must be Pandas Data frame or Series")

    
def Portfolio_returns(weights, r):
    return weights.T @ r


def Portfolio_volatility(weights, cov_mat):
    return (weights.T @ cov_mat @ weights)**0.5


def Minimize_vol(target_ret, rets, cov):
    """
    Returns minimum volatility weights for a given target return
    """
    n = rets.shape[0]
    initial_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n
    match_target = {
        'type' : 'eq',
        'args' : (rets,),
        'fun': lambda weights, rets: target_ret - Portfolio_returns(weights, rets)
    }
    weights_sum_to_1 = {
        'type' : 'eq', 
        'fun' : lambda weights: np.sum(weights) - 1
    }
    results = minimize(Portfolio_volatility, initial_guess, args=(cov,), 
                       method = "SLSQP", bounds = bounds, 
                       constraints = (match_target, weights_sum_to_1))
    return results.x


def msr(rf, rets, cov):
    """
    Returns max sharpe ratio weights for a given risk free rate
    """
    n = rets.shape[0]
    initial_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n
   
    weights_sum_to_1 = {
        'type' : 'eq', 
        'fun' : lambda weights: np.sum(weights) - 1
    }
    
    def neg_sharpe_ratio(weights, rf, rets, cov):
        return -(Portfolio_returns(weights, rets) - rf)/Portfolio_volatility(weights, cov)
    results = minimize(neg_sharpe_ratio, initial_guess, args=(rf, rets, cov,), 
                       method = "SLSQP", bounds = bounds, 
                       constraints = (weights_sum_to_1))
    return results.x


def gmv_weights(cov):
    n_assets = cov.shape[0]
    return msr(0, np.repeat(1, n_assets), cov)


def Efficient_frontier(n, rets, cov, cml=False, rf=0.0, ew=False, gmv=False, line_style = '.-'):
    """
    Plots Efficient Frontier. show_cml = True, gives the CML as well.
    """
    target_rs = np.linspace(rets.min(), rets.max(), n)
    weights = [Minimize_vol(target_rets, rets, cov) for target_rets in target_rs]
    Returns = [Portfolio_returns(w, rets) for w in weights ]
    Volatilities = [Portfolio_volatility(w, cov) for w in weights ]
    
    efficient_frontier = pd.DataFrame({
        "Returns" : Returns,
        "Volatilities" : Volatilities
    })
    ax = efficient_frontier.plot.line(x ="Volatilities", y = "Returns", style = line_style)
    ax.set_xlim(0)
   
    if cml:
        msr_w = msr(rf, rets, cov)
        msr_r = Portfolio_returns(msr_w, rets)
        msr_v = Portfolio_volatility(msr_w, cov)
        #Add CML
        x = [0, msr_v]
        y = [rf, msr_r]
        ax.plot(x, y, color = "green", marker ="o")
        ax.text(msr_v-0.001, msr_r, "MSR", ha='right', va = 'bottom', color='green', fontsize=8)
        
    if ew:
        n_assets = rets.shape[0]
        ew_w = np.repeat(1/n_assets, n_assets)
        ew_r = Portfolio_returns(ew_w, rets)
        ew_v = Portfolio_volatility(ew_w, cov)
        #EW portfolio
        ax.plot([ew_v], [ew_r], marker = "o", color="grey")
        ax.text(ew_v + 0.002, ew_r, "EW", va = 'center', color='grey', fontsize=8)
        
    if gmv:
        gmv_w = gmv_weights(cov)
        gmv_r = Portfolio_returns(gmv_w, rets)
        gmv_v = Portfolio_volatility(gmv_w, cov)
        #EW portfolio
        ax.plot([gmv_v], [gmv_r], marker = "o", color="midnightblue")
        ax.text(gmv_v - 0.002, gmv_r, "GMV", ha='right', va = 'center', color='midnightblue', fontsize=8)
        
    return ax