import pandas as pd
import numpy as np
import scipy.stats
import matplotlib as plt
from scipy.stats import norm
from scipy.optimize import minimize
import math

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


def Sharpe_ratio(returns, rf, period):
    """
    INPUTS
    Return series/ dataframe
    Annualized Risk free rate in decimal
    no. of period in a year (12 for monthly, 252 for daily)
    """
    period_rf = (1+rf)**(1/period) -1
    excess_returns = returns - period_rf
    ann_excess_returns = Returns_ann(excess_returns, period)
    return ann_excess_returns/Vola_ann(returns, period)


def Drawdown (return_series: pd.Series):
    """
    Input Dataframe and returns a dataframe with wealthindex, previouspeak and drawdown
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
        shortfall = r <= -VaR_historic(r, level=level)
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
        'type' : 'eq',  #means function must be equal to 0
        'args' : (rets,), # states the non-changing argument, in this case the optimizer is only allowed to change "weights" and not "rets"
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


def CPPI(risky_r, safe_r=None, m=3, start=1000, floor=0.8, rf = 0.03, drawdown = None):
    """
    Run a backtest of CPPI strategy, given a set of returns for risky asset.
    Returns a dictionary of History containing: Asset value, Risky budget, Risky weight
    """
    #setup parameters
    dates = risky_r.index
    n_steps = len(dates)
    acc_val = start
    flr_val = start * floor
    peak = start
    
    if isinstance(risky_r, pd.Series):
        risky_r = pd.DataFrame(risky_r, columns["R"])
        
    if safe_r is None:
        safe_r = pd.DataFrame().reindex_like(risky_r)
        safe_r.values[:] = rf/12
        
    #setup Df for saving history
    acc_hist = pd.DataFrame().reindex_like(risky_r)
    risky_w_hist = pd.DataFrame().reindex_like(risky_r)
    cushion_hist = pd.DataFrame().reindex_like(risky_r)
    
    for step in range(n_steps):
        if drawdown is not None:
            peak = np.maximum(peak, acc_val)
            flr_val = peak*(1-drawdown)
        cushion = (acc_val - flr_val)/acc_val
        risky_w = m*cushion
        risky_w = np.minimum(risky_w, 1)
        risky_w = np.maximum(risky_w, 0)
        safe_w = 1-risky_w
        risky_alloc = risky_w*acc_val
        safe_alloc = safe_w*acc_val
        #new account value
        acc_val = risky_alloc*(1+risky_r.iloc[step]) + safe_alloc*(1+safe_r.iloc[step])
        #save step history
        acc_hist.iloc[step] = acc_val
        risky_w_hist.iloc[step] = risky_w
        cushion_hist.iloc[step] = cushion
        
    risky_wealth = start*(1+risky_r).cumprod()
    backtest = {
        "Wealth": acc_hist,
        "Risky Wealth": risky_wealth,
        "Risk Budget": cushion_hist,
        "Risk Allocation": risky_w_hist,
        "Floor": flr_val
        
    }
    return backtest


def Summary_stats(r, rf=0.03):
    """
    Return a summary of stats for all the columns in the returns df r
    """
    ann_r = r.aggregate(Returns_ann, period=12)
    ann_v = r.aggregate(Vola_ann, period = 12)
    ann_sr = r.aggregate(Sharpe_ratio, rf=rf, period=12)
    dd = r.aggregate(lambda r: Drawdown(r).Drawdowns.min())
    sk = r.aggregate(Skewness)
    kr = r.aggregate(Kurtosis)
    cf_var5 = r.aggregate(VaR_gaussian, modified=True)
    hist_cvar = r.aggregate(cVaR_historic)
    
    return pd.DataFrame({
        "Annualized Return": ann_r,
        "Annualized Volatility": ann_v,
        "Skewness": sk,
        "Kurtosis": kr,
        "Cornish-Fisher VaR(5%)":cf_var5,
        "cVaR Historic (5%)": hist_cvar,
        "Sharpe Ratio": ann_sr,
        "Max Drawdown": dd
    })


def Gbm(n_years=10, n_scenarios=1000, steps_per_year=12, mu=0.07, sigma=0.15, s0=100, prices=True):
    """
    Evolution of Geometric Brownian Motion trajectories, such as for Stock Prices through Monte Carlo
    :param n_years:  The number of years to generate data for
    :param n_paths: The number of scenarios/trajectories
    :param mu: Annualized Drift, e.g. Market Return
    :param sigma: Annualized Volatility
    :param steps_per_year: granularity of the simulation
    :param s_0: initial value
    :return: a numpy array of n_paths columns and n_years*steps_per_year rows
    """
    dt= 1/steps_per_year
    n_steps = int(n_years * steps_per_year) + 1
    rets_plus_1 = np.random.normal(loc=(mu*dt)+1, scale=(sigma*np.sqrt(dt)), size=(n_steps,n_scenarios))
    rets_plus_1[0] = 1
    
    ret_val = s0*pd.DataFrame(rets_plus_1).cumprod() if prices else rets_plus_1-1
    return ret_val


def discount(t, r):
    """
    Compute the price of a pure discount bond that pays a dollar at time t where t is in years and r is the annual interest rate
    """
    return (1+r)**(-t)


def pv(l, r):
    """
    Compute the present value of a list of liabilities given by the time (as an index) and amounts
    """
    dates = l.index
    discounts = discount(dates, r)
    return (discounts*l).sum()


def funding_ratio(assets, liabilities, r):
    """
    Computes the funding ratio of a series of liabilities, based on an interest rate and current value of assets
    """
    return assets/pv(liabilities, r)


def inst_to_ann(r):
    """
    Convert an instantaneous interest rate to an annual interest rate
    """
    return np.expm1(r)


def ann_to_inst(r):
    """
    Convert an instantaneous interest rate to an annual interest rate
    """
    return np.log1p(r)


def cir(n_years = 10, n_scenarios=1, a=0.05, b=0.03, sigma=0.05, steps_per_year=12, r_0=None):
    """
    Generate random interest rate evolution over time using the CIR model
    b and r_0 are assumed to be the annualized rates, not the short rate
    and the returned values are the annualized rates as well
    """
    if r_0 is None: r_0 = b 
    r_0 = ann_to_inst(r_0)
    dt = 1/steps_per_year
    num_steps = int(n_years*steps_per_year) + 1 # because n_years might be a float
    
    shock = np.random.normal(0, scale=np.sqrt(dt), size=(num_steps, n_scenarios))
    rates = np.empty_like(shock)
    rates[0] = r_0

    # For Price Generation of zero coupon : Refer Notes for Price 
    h = math.sqrt(a**2 + 2*sigma**2)
    prices = np.empty_like(shock)
    

    def price(ttm, r):
        if sigma == 0:
        # Simplified formulas for the deterministic case
            _B = (1 - np.exp(-a * ttm)) / a
            _A = np.exp(b * (_B - ttm))
        else:
            _A = ((2*h*math.exp((h+a)*ttm/2))/(2*h+(h+a)*(math.exp(h*ttm)-1)))**(2*a*b/sigma**2)
            _B = (2*(math.exp(h*ttm)-1))/(2*h + (h+a)*(math.exp(h*ttm)-1))
        _P = _A*np.exp(-_B*r)
        return _P
    prices[0] = price(n_years, r_0)

    
    for step in range(1, num_steps):
        r_t = rates[step-1]
        d_r_t = a*(b-r_t)*dt + sigma*np.sqrt(r_t)*shock[step]
        rates[step] = abs(r_t + d_r_t)
        # generate prices at time t as well ...
        prices[step] = price(n_years-step*dt, rates[step])

    rates = pd.DataFrame(data=inst_to_ann(rates), index=range(num_steps))
    # for prices
    prices = pd.DataFrame(data=prices, index=range(num_steps))

    return rates, prices


def bond_cashflows(maturity, princpal = 100, coupon_rate=0.03, coupon_per_year=12 ):
    """
    Returns the series of cash flows generated by a bond,
    indexed by the payment/coupon number
    """
    n_coupon = round(maturity*coupon_per_year)
    coupon = princpal*coupon_rate/coupon_per_year
    cashflows = np.repeat(coupon, n_coupon)
    n_times = np.arrange(1, n_coupon+1)
    bond_cashflows = pd.Series(data = cashflows, index=n_times)
    bond_cashflows.iloc[-1] += princpal
    return bond_cashflows


def bond_price(maturity, pricipal, coupon_rate, coupon_per_year, rf=0.05):
    """
    Computes the price of a bond that pays regular coupons until maturity
    at which time the principal and the final coupon is returned
    This is not designed to be efficient, rather,
    it is to illustrate the underlying principle behind bond pricing!
    """
    cashflows = bond_cashflows(maturity, pricipal, coupon_rate, coupon_per_year)
    return pv(cashflows, rf/coupon_per_year)


def macaulay_duration(cashflows, rf):
    """
    Computes the Macaulay Duration of a sequence of cash flows, given a per-period discount rate
    """
    discounted_cflows = discount(cashflows.index, rf)*cashflows
    weights = discounted_cflows/discounted_cflows.sum()
    return np.average(cashflows.index, weights=weights)

def match_durations(cf_t, cf_s, cf_l, discount_rate):
    """
    Returns the weight W in cf_s that, along with (1-W) in cf_l will have an effective
    duration that matches cf_t
    """
    #Underlying Theory: The duration of a portfolio is the weighted avg of the asset durations
    d_t = macaulay_duration(cf_t, discount_rate)
    d_s = macaulay_duration(cf_s, discount_rate)
    d_l = macaulay_duration(cf_l, discount_rate)
    return (d_l - d_t)/(d_l - d_s)
