"""
Author:     Aaron Niecestro
Project:    Deep Learning for Wine Quality (DLWQ)

Created:    November 2, 2026
Last Edit:  November 2, 2026
Progress:   Complete

Analysis:   08 - Final Model Selection
Method:     Final Recommendation Script

Description:
Documents the final recommended neural network
configuration identified through the complete
DLWQ experimental process.

Research Question:
What is the strongest-performing neural network
configuration for wine quality prediction?
"""

# ==========================================
# Imports
# ==========================================

import pandas as pd

from pathlib import Path

# ==========================================
# Directories
# ==========================================

analysis07_dir = Path(
    r"C:\Users\aniec\Desktop\ibm-ml-project\deep-learning-rl\08-Regularization-and-Generalization\tables_figures"
)

output_dir = Path(
    r"C:\Users\aniec\Desktop\ibm-ml-project\deep-learning-rl\09-Final-Model-Selection\tables_figures"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

# ==========================================
# Winning Configuration
# ==========================================

winning_configuration = pd.DataFrame({

    "Component": [
        "Model Type",
        "Scaling Method",
        "Architecture",
        "Activation Function",
        "Optimizer",
        "Epochs",
        "Regularization"
    ],

    "Winner": [
        "MLP",
        "StandardScaler",
        "64-32-16",
        "Sigmoid",
        "RMSProp",
        "125",
        "None (Baseline)"
    ]

})

winning_configuration.to_csv(

    output_dir /
    "winning_configuration.csv",

    index=False

)

# ==========================================
# Load Analysis 07 Results
# ==========================================

comparison = pd.read_csv(

    analysis07_dir /
    "regularization_comparison.csv"

)

comparison = comparison.sort_values(
    by="RMSE"
)

winner = comparison.iloc[0]

best_method = winner["Method"]

print(
    f"\nWinning Method: {best_method}"
)

# ==========================================
# Load Winning Results File
# ==========================================

final_results = pd.read_csv(

    analysis07_dir /
    f"{best_method.lower()}_results.csv"

)

# ==========================================
# Export Final Results
# ==========================================

final_results.to_csv(

    output_dir /
    "final_model_results.csv",

    index=False

)

# ==========================================
# Summary Report
# ==========================================

with open(

    output_dir /
    "final_model_summary.txt",

    "w",

    encoding="utf-8"

) as f:

    f.write(
        "DLWQ Final Model Recommendation\n"
    )

    f.write(
        "=" * 60
    )

    f.write(
        "\n\nWinning Configuration\n\n"
    )

    f.write(
        winning_configuration
        .to_string(index=False)
    )

    f.write(
        "\n\nWinning Regularization Method\n\n"
    )

    f.write(
        f"{best_method}\n"
    )

    f.write(
        "\nFinal Performance Metrics\n\n"
    )

    f.write(
        final_results
        .round(4)
        .to_string(index=False)
    )
    
    f.write(
        "\n\nFinal Pipeline\n\n"
    )

    f.write(
        "MLP\n"
    )

    f.write(
        "-> StandardScaler\n"
    )

    f.write(
        "-> Sigmoid\n"
    )

    f.write(
        "-> Architecture (64 -> 32 -> 16)\n"
    )

    f.write(
        "-> 125 Epochs\n"
    )

    f.write(
        "-> RMSProp\n"
    )

    f.write(
        "-> No Regularization (Baseline)\n"
    )
    
    f.write(
        "\n\nModel Selection Summary\n"
    )

    f.write(
        f"\nWinning Method: {best_method}"
    ) 

# ==========================================
# Console Output
# ==========================================

print(
    "\n----- FINAL MODEL RECOMMENDATION -----"
)

print(
    winning_configuration.to_string(
        index=False
    )
)

print(
    "\n----- FINAL PERFORMANCE -----"
)

print(
    f"Method: {final_results.iloc[0]['Method']}"
)

print(
    f"MAE: {final_results.iloc[0]['MAE']:.4f}"
)

print(
    f"RMSE: {final_results.iloc[0]['RMSE']:.4f}"
)

print(
    f"Adjusted R²: "
    f"{final_results.iloc[0]['Adjusted_R2']:.4f}"
)

print(
    f"Training Time: "
    f"{final_results.iloc[0]['Training_Time_Seconds']:.2f} seconds"
)

print(
    f"Final Training Loss: "
    f"{final_results.iloc[0]['Final_Training_Loss']:.4f}"
)

print(
    f"Final Validation Loss: "
    f"{final_results.iloc[0]['Final_Validation_Loss']:.4f}"
)

print(
    f"Generalization Gap: "
    f"{final_results.iloc[0]['Generalization_Gap']:.4f}"
)

print(
    "\n----- FINAL PIPELINE -----"
)

print(
    f"\nWinning Method: {best_method}"
)

print(
    "MLP"
)

print(
    "-> StandardScaler"
)

print(
    "-> Sigmoid"
)

print(
    "-> 64 -> 32 -> 16"
)

print(
    "-> 125 Epochs"
)

print(
    "-> RMSProp"
)

print(
    "-> No Regularization (Baseline)"
)

print(
    "\nAnalysis 09 Complete"
)