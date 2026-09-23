# -*- coding: utf-8 -*-
"""
Author:     Aaron Niecestro
Project:    DS Salaries – Production-Ready EDA for Regression Models

Created:    September 17 ,2026
Last Edit:  September 23, 2026
Progress:   Completed

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
    - Unified Leaderboard — DS Salaries Regression Models
"""

# =========================================================
# Imports: Pull Best Metrics From Each Model File
# =========================================================

import pandas as pd

# Linear Regression Log Transformation (OLS)
from DSS_01a_linear_reg_log_trans import (
    best_ols_rmse,
    best_ols_mae,
    best_ols_r2,
    best_ols_adj_r2
)

# Polynomial / Interaction Models
from DSS_02_poly_reg import (
    best_poly_rmse,
    best_poly_mae,
    best_poly_r2,
    best_poly_adj_r2
)

from DSS_02_poly_reg import (
    best_inter_rmse,
    best_inter_mae,
    best_inter_r2,
    best_inter_adj_r2
)

from DSS_02_poly_reg import (
    best_poly_inter_rmse,
    best_poly_inter_mae,
    best_poly_inter_r2,
    best_poly_inter_adj_r2
)

# Regularization Models
from DSS_03_ridge_lasso_net import (
    best_ridge_rmse,
    best_ridge_mae,
    best_ridge_r2,
    best_ridge_adj_r2
)

from DSS_03_ridge_lasso_net import (
    best_lasso_rmse,
    best_lasso_mae,
    best_lasso_r2,
    best_lasso_adj_r2
)

from DSS_03_ridge_lasso_net import (
    best_elastic_rmse,
    best_elastic_mae,
    best_elastic_r2,
    best_elastic_adj_r2
)

# Tree-Based Models
from DSS_04_reg_trees import (
    best_rf_rmse,
    best_rf_mae,
    best_rf_r2,
    best_rf_adj_r2
)

from DSS_04_reg_trees import (
    best_xgb_rmse,
    best_xgb_mae,
    best_xgb_r2,
    best_xgb_adj_r2
)

from DSS_04_reg_trees import (
    best_lgbm_rmse,
    best_lgbm_mae,
    best_lgbm_r2,
    best_lgbm_adj_r2
)

# Support Vector Regression
from DSS_05_svm import (
    best_svr_rmse,
    best_svr_mae,
    best_svr_r2,
    best_svr_adj_r2
)

# =========================================================
# Build Unified Leaderboard Table
# =========================================================

leaderboard = pd.DataFrame({
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
        best_ols_rmse,
        best_poly_rmse,
        best_inter_rmse,
        best_poly_inter_rmse,
        best_ridge_rmse,
        best_lasso_rmse,
        best_elastic_rmse,
        best_rf_rmse,
        best_xgb_rmse,
        best_lgbm_rmse,
        best_svr_rmse
    ],
    "MAE": [
        best_ols_mae,
        best_poly_mae,
        best_inter_mae,
        best_poly_inter_mae,
        best_ridge_mae,
        best_lasso_mae,
        best_elastic_mae,
        best_rf_mae,
        best_xgb_mae,
        best_lgbm_mae,
        best_svr_mae
    ],
    "R^2": [
        best_ols_r2,
        best_poly_r2,
        best_inter_r2,
        best_poly_inter_r2,
        best_ridge_r2,
        best_lasso_r2,
        best_elastic_r2,
        best_rf_r2,
        best_xgb_r2,
        best_lgbm_r2,
        best_svr_r2
    ],
    "Adj R^2": [
    best_ols_adj_r2,
    best_poly_adj_r2,
    best_inter_adj_r2,
    best_poly_inter_adj_r2,
    best_ridge_adj_r2,
    best_lasso_adj_r2,
    best_elastic_adj_r2,
    best_rf_adj_r2,
    best_xgb_adj_r2,
    best_lgbm_adj_r2,
    best_svr_adj_r2
    ]
})


# =========================================================
# Sort Leaderboard by RMSE (Lower = Better)
# =========================================================

leaderboard = leaderboard.sort_values("RMSE").reset_index(drop=True)

print("\n===== UNIFIED MODEL LEADERBOARD (ALL REGRESSION FAMILIES) =====")
print(leaderboard)
