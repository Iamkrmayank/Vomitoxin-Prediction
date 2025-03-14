# Installation Guide and Repository Structure

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
