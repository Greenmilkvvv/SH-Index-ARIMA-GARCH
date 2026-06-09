# models/garch_models.py —— GARCH(1,1)-Normal 和 EGARCH(1,1)-t 模型

import pandas as pd
import numpy as np
from arch import arch_model


def fit_garch_normal(residuals: pd.Series) -> dict:
    """拟合 GARCH(1,1)-Normal 模型.

    Parameters
    ----------
    residuals : pd.Series
        ARIMA 残差序列（训练集）

    Returns
    -------
    dict
        包含：
        - model: arch 拟合对象
        - params: 参数估计表 (pd.DataFrame)
        - log_likelihood: 对数似然值
        - aic: AIC
        - bic: BIC
        - conditional_volatility: 条件波动率序列
    """
    # rescale 以提高数值稳定性
    resid = residuals.dropna()
    scale = 1.0
    if resid.std() < 0.01:
        scale = 100.0
    scaled = resid * scale

    model = arch_model(scaled, mean="Zero", vol="GARCH", p=1, q=1, dist="normal")
    fitted = model.fit(disp="off")

    # 参数表
    params_df = pd.DataFrame(
        {
            "参数名": fitted.params.index,
            "估计值": fitted.params.values,
            "标准误": fitted.std_err.values,
            "t值": fitted.tvalues.values,
            "P值": fitted.pvalues.values,
        }
    ).reset_index(drop=True)

    # 条件波动率还原缩放
    cond_vol = fitted.conditional_volatility / scale

    return {
        "model": fitted,
        "params": params_df,
        "log_likelihood": fitted.loglikelihood,
        "aic": fitted.aic,
        "bic": fitted.bic,
        "conditional_volatility": cond_vol,
    }


def fit_egarch_t(residuals: pd.Series) -> dict:
    """拟合 EGARCH(1,1)-t 模型.

    Parameters
    ----------
    residuals : pd.Series
        ARIMA 残差序列（训练集）

    Returns
    -------
    dict
        包含：
        - model: arch 拟合对象
        - params: 参数估计表 (pd.DataFrame)
        - log_likelihood: 对数似然值
        - aic: AIC
        - bic: BIC
        - conditional_volatility: 条件波动率序列
    """
    resid = residuals.dropna()
    scale = 1.0
    if resid.std() < 0.01:
        scale = 100.0
    scaled = resid * scale

    model = arch_model(scaled, mean="Zero", vol="EGARCH", p=1, q=1, dist="t")
    fitted = model.fit(disp="off")

    params_df = pd.DataFrame(
        {
            "参数名": fitted.params.index,
            "估计值": fitted.params.values,
            "标准误": fitted.std_err.values,
            "t值": fitted.tvalues.values,
            "P值": fitted.pvalues.values,
        }
    ).reset_index(drop=True)

    cond_vol = fitted.conditional_volatility / scale

    return {
        "model": fitted,
        "params": params_df,
        "log_likelihood": fitted.loglikelihood,
        "aic": fitted.aic,
        "bic": fitted.bic,
        "conditional_volatility": cond_vol,
    }