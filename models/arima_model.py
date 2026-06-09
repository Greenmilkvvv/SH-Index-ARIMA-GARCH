# models/arima_model.py —— ARIMA 模型拟合与信息提取

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from config import ARIMA_ORDER


def fit_arima(train_series: pd.Series) -> dict:
    """拟合 ARIMA 模型并返回结果.

    Parameters
    ----------
    train_series : pd.Series
        训练集对数收益率

    Returns
    -------
    dict
        包含以下键：
        - model: statsmodels ARIMA 拟合对象
        - residuals: 残差序列 (pd.Series)
        - aic: AIC 值
        - bic: BIC 值
        - log_likelihood: 对数似然值
        - r_squared: 伪 R² (1 - var(resid)/var(y))
        - params: 参数估计表 (pd.DataFrame)
    """
    order = ARIMA_ORDER
    model = ARIMA(train_series, order=order)
    fitted = model.fit()

    # 残差
    residuals = fitted.resid

    # 信息准则
    aic = fitted.aic
    bic = fitted.bic
    log_likelihood = fitted.llf

    # 伪 R²
    var_y = np.var(train_series)
    var_resid = np.var(residuals)
    r_squared = 1 - var_resid / var_y if var_y > 0 else np.nan

    # 参数表
    params_df = pd.DataFrame(
        {
            "参数名": fitted.params.index,
            "估计值": fitted.params.values,
            "标准误": fitted.bse.values,
            "z值": fitted.tvalues.values,
            "P值": fitted.pvalues.values,
        }
    ).reset_index(drop=True)

    return {
        "model": fitted,
        "residuals": residuals,
        "aic": aic,
        "bic": bic,
        "log_likelihood": log_likelihood,
        "r_squared": r_squared,
        "params": params_df,
        "n_obs": len(train_series),
        "df_resid": fitted.df_resid,
    }