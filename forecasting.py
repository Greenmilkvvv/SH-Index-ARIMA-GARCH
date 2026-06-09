# forecasting.py —— 滚动一步预测引擎

import numpy as np
import pandas as pd
from arch import arch_model
from config import ARIMA_ORDER


def rolling_arima_forecast(
    train_series: pd.Series,
    test_series: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """用 ARIMA 对测试集做滚动一步预测，返回残差序列.

    策略：使用训练集拟合 ARIMA，然后对测试集每一天做一步向前预测，
    记录预测残差（实际值 - 预测值），并用新的实际观测值扩展训练窗口。

    Parameters
    ----------
    train_series : pd.Series
        训练集对数收益率
    test_series : pd.Series
        测试集对数收益率

    Returns
    -------
    residuals_test : np.ndarray
        测试集上的 ARIMA 残差（预测残差）
    arima_forecasts : np.ndarray
        ARIMA 预测值
    """
    n_test = len(test_series)

    # 用全量训练数据初始拟合
    from statsmodels.tsa.arima.model import ARIMA

    try:
        model = ARIMA(train_series, order=ARIMA_ORDER)
        fitted = model.fit()
    except Exception:
        # 如果 MLE 不收敛，改用 CSS
        model = ARIMA(train_series, order=ARIMA_ORDER)
        fitted = model.fit(method="css")

    # 扩展序列：从训练数据开始
    history = list(train_series.values)
    forecasts = np.zeros(n_test)
    residuals_test = np.zeros(n_test)

    for i in range(n_test):
        # 对 t+i 时刻做一步向前预测
        try:
            model = ARIMA(history, order=ARIMA_ORDER)
            fitted = model.fit()
            fc = fitted.forecast(steps=1)
            forecasts[i] = fc[0]
        except Exception:
            forecasts[i] = np.mean(history[-20:]) if len(history) >= 20 else np.mean(history)

        actual = test_series.iloc[i]
        residuals_test[i] = actual - forecasts[i]

        # 用实际观测值扩展历史
        history.append(actual)

    return residuals_test, forecasts


def rolling_garch_forecast(
    train_residuals: pd.Series,
    test_residuals: np.ndarray,
    vol: str = "GARCH",
    dist: str = "normal",
) -> np.ndarray:
    """用 GARCH 族模型对测试集残差做滚动一步条件方差预测.

    Parameters
    ----------
    train_residuals : pd.Series
        训练集 ARIMA 残差
    test_residuals : np.ndarray
        测试集 ARIMA 残差 (由 rolling_arima_forecast 产出)
    vol : str
        波动率模型类型 ("GARCH" 或 "EGARCH")
    dist : str
        误差分布类型 ("normal" 或 "t")

    Returns
    -------
    cond_var_forecasts : np.ndarray
        测试集的条件方差预测值序列
    """
    resid = train_residuals.dropna()
    scale = 1.0
    if resid.std() < 0.01:
        scale = 100.0
    scaled_train = resid.values * scale

    # 初始拟合
    model = arch_model(scaled_train, mean="Zero", vol=vol, p=1, q=1, dist=dist)
    fitted = model.fit(disp="off")

    # 获取训练集最后一期的条件方差作为初始值
    train_cond_var = fitted.conditional_volatility[-1] / scale

    n_test = len(test_residuals)
    cond_var_forecasts = np.zeros(n_test)

    # 滚动参数
    window = list(scaled_train)  # 以缩放后的残差作为历史
    current_var = fitted.conditional_volatility[-1]  # 上一期条件方差（缩放尺度）

    for i in range(n_test):
        # GARCH(1,1): sigma²_t = omega + alpha * eps²_{t-1} + beta * sigma²_{t-1}
        # EGARCH(1,1): ln(sigma²_t) = omega + beta * ln(sigma²_{t-1})
        #                                + alpha * (|z_{t-1}| - E[|Z|]) + gamma * z_{t-1}
        params = fitted.params

        if i == 0:
            last_eps_scaled = window[-1]  # 上一期残差（缩放）
        else:
            last_eps_scaled = test_residuals[i - 1] * scale

        if vol == "GARCH":
            omega = params.get("omega", 0)
            alpha = params.get("alpha[1]", 0)
            beta = params.get("beta[1]", 0)
            # 条件方差（缩放尺度）
            new_var_scaled = omega + alpha * (last_eps_scaled ** 2) + beta * current_var
            current_var = max(new_var_scaled, 1e-12)
            cond_var_forecasts[i] = current_var / (scale ** 2)  # 还原
        elif vol == "EGARCH":
            omega = params.get("omega", 0)
            beta = params.get("beta[1]", 0)
            alpha = params.get("alpha[1]", 0)
            gamma = params.get("gamma[1]", 0)
            # z_{t-1} = eps_{t-1} / sigma_{t-1}
            sigma_prev_scaled = np.sqrt(max(current_var, 1e-12))
            z = last_eps_scaled / sigma_prev_scaled if sigma_prev_scaled > 1e-12 else 0

            # E[|Z|] under t-dist: 2 * sqrt(nu-2) * Gamma((nu+1)/2) / (sqrt(pi) * Gamma(nu/2) * (nu-1))
            from scipy.special import gammaln
            if dist == "t":
                nu = params.get("nu", 10)
                if nu > 4:  # 防止数值问题
                    e_abs_z = (
                        2.0
                        * np.sqrt(nu - 2)
                        * np.exp(gammaln((nu + 1) / 2) - gammaln(nu / 2))
                        / (np.sqrt(np.pi) * (nu - 1))
                    )
                else:
                    e_abs_z = np.sqrt(2 / np.pi)  # 退化为正态
            else:
                e_abs_z = np.sqrt(2 / np.pi)

            log_var_new = omega + beta * np.log(max(current_var, 1e-12)) + alpha * (abs(z) - e_abs_z) + gamma * z
            new_var_scaled = np.exp(log_var_new)
            current_var = max(new_var_scaled, 1e-12)
            cond_var_forecasts[i] = current_var / (scale ** 2)

        # 用实际值更新历史并(可选)重新估计——论文采用预训练参数不复现
        # 为保持一致性，每步重新估计模型
        window.append(test_residuals[i] * scale)
        # 每 20 步重新估计一次以平衡速度和精度
        if (i + 1) % 20 == 0:
            try:
                model = arch_model(np.array(window), mean="Zero", vol=vol, p=1, q=1, dist=dist)
                fitted = model.fit(disp="off")
            except Exception:
                pass  # 保持原参数

    return cond_var_forecasts


def compute_forecast_metrics(
    actual_vol: np.ndarray,
    forecast_vol: np.ndarray,
) -> dict:
    """计算预测评价指标.

    Parameters
    ----------
    actual_vol : np.ndarray
        实际波动率（代理：残差平方）
    forecast_vol : np.ndarray
        预测的条件方差

    Returns
    -------
    dict
        MSE, RMSE, MAE, QLIKE
    """
    eps = 1e-12
    actual = np.maximum(actual_vol, eps)
    forecast = np.maximum(forecast_vol, eps)

    errors = actual - forecast
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(errors))

    # QLIKE: actual/forecast - log(actual/forecast) - 1 的均值
    ratio = actual / forecast
    qlike = np.mean(ratio - np.log(ratio) - 1)

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "QLIKE": qlike,
    }