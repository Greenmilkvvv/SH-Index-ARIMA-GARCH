# main.py —— 主控脚本，一键运行全流程
# 用法: py -3.10 main.py

import sys
import numpy as np
import pandas as pd

# 本地模块
from config import FIGURES_DIR
import data_loader
import diagnostics
from models import fit_arima, fit_garch_normal, fit_egarch_t
import forecasting
import visualization
import report_generator


def format_p(p_val: float, use_stars: bool = True) -> str:
    """格式化 p 值显示."""
    if use_stars:
        if p_val < 0.01:
            return f"{p_val:.4f}***"
        elif p_val < 0.05:
            return f"{p_val:.4f}**"
        elif p_val < 0.10:
            return f"{p_val:.4f}*"
    return f"{p_val:.4f}"


def print_separator(title: str):
    """打印分隔标题."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    import matplotlib.pyplot as plt

    # ==================== 1. 加载数据 ====================
    print_separator("1. 数据加载与划分")
    full_series, train_series, test_series = data_loader.load_data()
    print(f"  总样本: {len(full_series)} 个交易日 ({full_series.index[0].date()} 至 {full_series.index[-1].date()})")
    print(f"  训练集: {len(train_series)} 个交易日 ({train_series.index[0].date()} 至 {train_series.index[-1].date()})")
    print(f"  测试集: {len(test_series)} 个交易日 ({test_series.index[0].date()} 至 {test_series.index[-1].date()})")

    # ==================== 2. Table 1: ADF 检验 ====================
    print_separator("2. Table 1 — ADF 单位根检验")
    adf_results = diagnostics.adf_test(full_series)
    print(adf_results.to_string(index=False))

    # ==================== 3. Figure 1: 对数收益率时间序列 ====================
    print_separator("3. Figure 1 — 对数收益率时间序列")
    visualization.plot_log_return_series(full_series)
    print(f"  已保存: result/figures/figure1_log_return.pdf")

    # ==================== 4. Figure 2: ACF / PACF ====================
    print_separator("4. Figure 2 — ACF / PACF 图")
    visualization.plot_acf_pacf(full_series)
    print(f"  已保存: result/figures/figure2_acf_pacf.pdf")

    # ==================== 5. Table 2: ARIMA(2,0,2) ====================
    print_separator("5. Table 2 — ARIMA(2,0,2) 模型拟合")
    arima_results = fit_arima(train_series)
    residuals_train = arima_results["residuals"]
    print(f"  AIC: {arima_results['aic']:.3f}")
    print(f"  BIC: {arima_results['bic']:.3f}")
    print(f"  Log-Likelihood: {arima_results['log_likelihood']:.3f}")
    print(f"  R²: {arima_results['r_squared']:.4f}")
    print(f"  样本量: {arima_results['n_obs']}")
    print(f"  Df Residuals: {arima_results['df_resid']}")
    print("\n  参数估计:")
    print(arima_results["params"].to_string(index=False))

    # Ljung-Box Q 检验
    print("\n  Ljung-Box Q 检验 (残差白噪声):")
    q_results = diagnostics.ljung_box_test(residuals_train, lags=[6, 12, 18, 24, 30])
    print(q_results.to_string(index=False))

    # ==================== 6. ARCH-LM 检验 ====================
    print_separator("6. ARCH-LM 检验 (残差异方差)")
    arch_lm_results = diagnostics.arch_lm_test(residuals_train)
    print(f"  LM 统计量: {arch_lm_results['LM统计量']:.4f}")
    print(f"  LM P 值:   {arch_lm_results['LM_P值']:.6f}")
    print(f"  F 统计量:  {arch_lm_results['F统计量']:.4f}")
    print(f"  F P 值:    {arch_lm_results['F_P值']:.6f}")
    print(f"  滞后阶数:  {arch_lm_results['滞后阶数']}")
    if arch_lm_results["LM_P值"] < 0.05:
        print("  ⇒ 残差存在显著的 ARCH 效应，需要引入 GARCH 族模型")
    else:
        print("  ⇒ 残差无显著 ARCH 效应")

    # ==================== 7. Table 3: GARCH & EGARCH ====================
    print_separator("7. Table 3 — GARCH(1,1)-Normal & EGARCH(1,1)-t 拟合")
    print("  正在拟合 GARCH(1,1)-Normal ...")
    garch_results = fit_garch_normal(residuals_train)
    print(f"    Log-Likelihood: {garch_results['log_likelihood']:.1f}")
    print(f"    AIC: {garch_results['aic']:.1f}")
    print(f"    BIC: {garch_results['bic']:.1f}")
    print("    参数:")
    print(garch_results["params"].to_string(index=False))

    print("\n  正在拟合 EGARCH(1,1)-t ...")
    egarch_results = fit_egarch_t(residuals_train)
    print(f"    Log-Likelihood: {egarch_results['log_likelihood']:.1f}")
    print(f"    AIC: {egarch_results['aic']:.1f}")
    print(f"    BIC: {egarch_results['bic']:.1f}")
    print("    参数:")
    print(egarch_results["params"].to_string(index=False))

    # ==================== 8. Table 4: 滚动预测与指标 ====================
    print_separator("8. Table 4 — 测试集滚动预测评估")

    # 8.1 ARIMA 滚动预测得到测试集残差
    print("  正在执行 ARIMA 滚动一步预测（测试集）...")
    arima_test_resid, arima_test_forecasts = forecasting.rolling_arima_forecast(
        train_series, test_series
    )
    # actual volatility proxy = squared residuals
    actual_vol = arima_test_resid ** 2

    # 8.2 GARCH 滚动预测
    print("  正在执行 GARCH 滚动波动率预测...")
    garch_vol_forecast = forecasting.rolling_garch_forecast(
        residuals_train, arima_test_resid, vol="GARCH", dist="normal"
    )
    garch_metrics = forecasting.compute_forecast_metrics(actual_vol, garch_vol_forecast)
    print(f"  GARCH  MSE:   {garch_metrics['MSE']:.6f}")
    print(f"  GARCH  RMSE:  {garch_metrics['RMSE']:.6f}")
    print(f"  GARCH  MAE:   {garch_metrics['MAE']:.6f}")
    print(f"  GARCH  QLIKE: {garch_metrics['QLIKE']:.6f}")

    # 8.3 EGARCH 滚动预测
    print("  正在执行 EGARCH 滚动波动率预测...")
    egarch_vol_forecast = forecasting.rolling_garch_forecast(
        residuals_train, arima_test_resid, vol="EGARCH", dist="t"
    )
    egarch_metrics = forecasting.compute_forecast_metrics(actual_vol, egarch_vol_forecast)
    print(f"  EGARCH MSE:   {egarch_metrics['MSE']:.6f}")
    print(f"  EGARCH RMSE:  {egarch_metrics['RMSE']:.6f}")
    print(f"  EGARCH MAE:   {egarch_metrics['MAE']:.6f}")
    print(f"  EGARCH QLIKE: {egarch_metrics['QLIKE']:.6f}")

    # ==================== 9. 生成报告 ====================
    print_separator("9. 生成报告")
    report_generator.generate_report(
        adf_df=adf_results,
        arima_results=arima_results,
        q_df=q_results,
        arch_lm=arch_lm_results,
        garch_results=garch_results,
        egarch_results=egarch_results,
        garch_metrics=garch_metrics,
        egarch_metrics=egarch_metrics,
        n_train=len(train_series),
        n_test=len(test_series),
        full_dates=(full_series.index[0], full_series.index[-1]),
        train_dates=(train_series.index[0], train_series.index[-1]),
        test_dates=(test_series.index[0], test_series.index[-1]),
    )
    print("  报告已保存: result/report.md")

    print_separator("完成")
    plt.close("all")


if __name__ == "__main__":
    main()