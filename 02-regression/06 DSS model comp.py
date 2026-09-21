# -*- coding: utf-8 -*-
"""
Author:     Aaron Niecestro
Project:    DS Salaries – Production-Ready EDA for Regression Models

Created:    September 17 ,2026
Last Edit:  September 21, 2026
Progress:   Ongoing
Author:     Aaron Niecestro
MASTER MODEL COMPARISON TABLE

Purpose:
    Compare the "best" models from all regression families:
        • OLS Linear Regression
        • Polynomial Regression (Degree 2)
        • Interaction-Only Model
        • Polynomial + Interaction Model
        • Regularization (Ridge, Lasso, Elastic Net)
        • Tree-Based Models (RF, XGB, LGBM)
        • Support Vector Regression (SVR, RBF Kernel)

Output:
    • Unified comparison table (RMSE, MAE, R²)
"""

# =========================================================
# Imports
# =========================================================

import numpy as np                                      # numerical ops
import pandas as pd                                     # data handling

# =========================================================
# Collect Best Model Metrics
# =========================================================
# NOTE:
# These values should come from your actual model runs.
# Replace each variable with the real metrics from your pipeline.

# ---------- Linear Regression ----------
rmse_ols = best_ols_rmse
mae_ols  = best_ols_mae
r2_ols   = best_ols_r2

# ---------- Polynomial Regression ----------
rmse_poly = best_poly_rmse
mae_poly  = best_poly_mae
r2_poly   = best_poly_r2

# ---------- Interaction-Only ----------
rmse_inter = best_inter_rmse
mae_inter  = best_inter_mae
r2_inter   = best_inter_r2

# ---------- Polynomial + Interaction ----------
rmse_poly_inter = best_poly_inter_rmse
mae_poly_inter  = best_poly_inter_mae
r2_poly_inter   = best_poly_inter_r2

# ---------- Regularization Models ----------
rmse_ridge = best_ridge_rmse
mae_ridge  = best_ridge_mae
r2_ridge   = best_ridge_r2

rmse_lasso = best_lasso_rmse
mae_lasso  = best_lasso_mae
r2_lasso   = best_lasso_r2

rmse_elastic = best_elastic_rmse
mae_elastic  = best_elastic_mae
r2_elastic   = best_elastic_r2

# ---------- Tree-Based Models ----------
rmse_rf = best_rf_rmse
mae_rf  = best_rf_mae
r2_rf   = best_rf_r2

rmse_xgb = best_xgb_rmse
mae_xgb  = best_xgb_mae
r2_xgb   = best_xgb_r2

rmse_lgbm = best_lgbm_rmse
mae_lgbm  = best_lgbm_mae
r2_lgbm   = best_lgbm_r2

# ---------- Support Vector Regression ----------
rmse_svr = best_svr_rmse
mae_svr  = best_svr_mae
r2_svr   = best_svr_r2

# =========================================================
# Build Comparison Table
# =========================================================

comparison_df = pd.DataFrame({
    "Model": [
        "OLS Linear Regression",
        "Polynomial Regression (Degree 2)",
        "Interaction-Only Model",
        "Polynomial + Interaction Model",
        "Ridge Regression",
        "Lasso Regression",
        "Elastic Net Regression",
        "Random Forest",
        "XGBoost",
        "LightGBM",
        "SVR (RBF Kernel)"
    ],
    "RMSE": [
        rmse_ols, rmse_poly, rmse_inter, rmse_poly_inter,
        rmse_ridge, rmse_lasso, rmse_elastic,
        rmse_rf, rmse_xgb, rmse_lgbm, rmse_svr
    ],
    "MAE": [
        mae_ols, mae_poly, mae_inter, mae_poly_inter,
        mae_ridge, mae_lasso, mae_elastic,
        mae_rf, mae_xgb, mae_lgbm, mae_svr
    ],
    "R²": [
        r2_ols, r2_poly, r2_inter, r2_poly_inter,
        r2_ridge, r2_lasso, r2_elastic,
        r2_rf, r2_xgb, r2_lgbm, r2_svr
    ]
})

# =========================================================
# Sort by RMSE (Best Model at Top)
# =========================================================

comparison_df = comparison_df.sort_values("RMSE")

print("\n===== FINAL MODEL COMPARISON TABLE =====")
print(comparison_df)
