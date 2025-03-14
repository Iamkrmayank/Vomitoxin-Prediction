## Results Summary & Installation Guide

## Results Summary

The analysis of vomitoxin prediction from spectral data yielded the following key results:

### Model Performance Comparison

| Model | RMSE | MAE | R² Score |
|-------|------|-----|----------|
| Random Forest | 2198.96 | 256.76 | 0.9827 |
| Gradient Boosting | 3170.04 | 391.03 | 0.9641 |
| XGBoost | 2612.54 | 350.33 | 0.9756 |
| Attention Model | 17318.59 | — | -0.0722 |
| Transformer Model | 17291.26 | — | -0.0692 |

### Key Findings

1. **Best Performing Model**: Random Forest demonstrated superior performance with the highest R² score (0.9827) and lowest error metrics (RMSE: 2198.96, MAE: 256.76).

2. **Traditional vs. Deep Learning Models**: Traditional machine learning models (Random Forest, Gradient Boosting, XGBoost) significantly outperformed deep learning approaches (Attention and Transformer models) for this specific task.

3. **Model Ranking**:
   - Random Forest: Best overall performance
   - XGBoost: Strong second-place performance
   - Gradient Boosting: Good performance
   - Attention and Transformer models: Poor performance with negative R² scores indicating worse predictions than a simple mean-based model

4. **Feature Importance**: 
   - The analysis identified several key spectral bands that showed strong predictive power for vomitoxin levels
   - Tree-based models provided valuable insights into the most influential spectral regions

5. **Preprocessing Impact**: 
   - Outlier handling through capping significantly improved model performance
   - Log transformation of the target variable helped address skewness issues

6. **Deep Learning Limitations**:
   - The negative R² scores for Attention and Transformer models suggest these architectures may not be suitable for this particular spectral dataset
   - Possible reasons include insufficient training data, overfitting, or the spectral features being better captured by tree-based ensemble methods

These results demonstrate that for this specific vomitoxin prediction task, traditional machine learning approaches, particularly Random Forest, provide the most accurate predictions from the available spectral data.


# Installation Guide 

## Dependencies Installation

### Prerequisites
- Python 3.8 or higher
- Pip package manager
- Google Colab (optional, for notebook execution)

### Method 1: Using pip (recommended for local execution)

```bash
# Clone the repository
git clone https://github.com/Iamkrmayank/Vomitoxin-Prediction.git
cd Vomitoxin-Prediction

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### Method 2: Using Google Colab (for notebook execution)

1. Upload the notebook `Task-updated-new.ipynb` to Google Colab
2. The notebook already includes all necessary pip installs with:
   ```python
   !pip install xgboost
   ```
3. For other dependencies, they are pre-installed in the Colab environment

## Running the Code

### Option 1: Jupyter Notebook / Google Colab
1. Open `Task-updated-new.ipynb` in Jupyter Notebook or Google Colab
2. Upload the dataset file `TASK-ML-INTERN.csv` when prompted
3. Run all cells sequentially using the "Run All" option or execute them individually

### Option 2: Python Script
1. Convert the notebook to a Python script (if not already available):
   ```bash
   jupyter nbconvert --to script Task-updated-new.ipynb
   ```
2. Run the script:
   ```bash
   python Task-updated-new.py
   ```

### Data Requirements
- Ensure the dataset file `TASK-ML-INTERN.csv` is available in the working directory
- The CSV file should contain columns for 'hsi_id', spectral bands, and 'vomitoxin_ppb'

## Key Files

- **Task-updated-new.ipynb**: Main notebook containing the complete analysis pipeline:
  - Data loading and exploration
  - Preprocessing and feature engineering
  - Dimensionality reduction with PCA
  - Model training with RandomForest, GradientBoosting, and XGBoost
  - Model evaluation and visualization
  
- **requirements.txt**: Contains all dependencies:
  ```
   streamlit
   pandas
   numpy
   matplotlib
   seaborn
   scikit-learn
   xgboost
  ```

## Troubleshooting

### Common Issues

1. **XGBoost Installation Errors**:
   - On Windows, you may need Microsoft Visual C++ Build Tools
   - Solution: Install the latest Microsoft C++ Build Tools or use a pre-compiled wheel

2. **Memory Issues**:
   - Large datasets may cause memory problems
   - Solution: Reduce the dataset size or use a system with more RAM

3. **Missing Data Files**:
   - Error: "FileNotFoundError: [Errno 2] No such file or directory: 'TASK-ML-INTERN.csv'"
   - Solution: Ensure the data file is in the correct location and has the correct name

