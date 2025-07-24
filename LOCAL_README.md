# Running Climate Prediction Model Locally

This guide explains how to run the climate temperature prediction model on your local machine to generate predictions, visualizations, and comprehensive reports.

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- 2GB available disk space
- Internet connection (for downloading climate data)

### Step 1: Setup Environment

```bash
# Clone the repository
git clone https://github.com/nishaero/ai-infra-samples.git
cd ai-infra-samples

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements/base.txt
```

### Step 2: Run the Model

```bash
# Basic run with default parameters (California region, 3 months of data)
python run_model.py

# The script will:
# 1. Download real climate data from NOAA
# 2. Train machine learning models
# 3. Generate predictions
# 4. Create visualization plots
# 5. Generate an HTML report
```

### Step 3: View Results

```bash
# Open the comprehensive HTML report
open output/climate_prediction_report.html  # macOS
start output/climate_prediction_report.html  # Windows
xdg-open output/climate_prediction_report.html  # Linux
```

## 📊 Output Explanation

After running the model, you'll get:

### 📄 HTML Report (`output/climate_prediction_report.html`)
A comprehensive report including:
- **Executive Summary**: High-level results overview
- **Performance Metrics**: MAE, RMSE, R² scores with explanations
- **Visualizations**: Interactive plots with detailed descriptions
- **Dataset Information**: Details about the data used
- **Methodology**: Explanation of the ML approach
- **Key Insights**: Actionable findings and recommendations

### 📈 Generated Plots (`output/plots/`)

1. **`actual_vs_predicted.png`**
   - Scatter plot comparing actual vs predicted temperatures
   - Perfect predictions would align on the diagonal line
   - Shows model accuracy and any systematic bias

2. **`residuals.png`**
   - Residual analysis showing prediction errors
   - Random distribution indicates good model performance
   - Patterns might suggest missing features or model bias

3. **`feature_importance.png`**
   - Bar chart showing which features contribute most to predictions
   - Helps understand what drives temperature variations
   - Useful for feature selection and model interpretation

4. **`model_comparison.png`**
   - Comparison of different algorithms (Linear vs Random Forest)
   - Shows strengths of ensemble approach
   - Helps validate model selection decisions

5. **`temperature_distribution.png`**
   - Histogram and box plot comparing actual vs predicted distributions
   - Ensures model captures the data distribution properly
   - Identifies any distribution shift issues

## ⚙️ Advanced Usage

### Custom Parameters

```bash
# Different geographic region (Pacific Northwest)
python run_model.py --bbox -130 42 -120 49

# Different time period (6 months)
python run_model.py --start-date 2023-01-01 --end-date 2023-06-30

# Custom output directory
python run_model.py --output-dir ./my_results

# Different train/test split
python run_model.py --test-size 0.3

# Complete custom run
python run_model.py \
    --start-date 2023-01-01 \
    --end-date 2023-12-31 \
    --bbox -125 30 -110 45 \
    --output-dir ./annual_results \
    --test-size 0.2
```

### Parameter Descriptions

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--start-date` | Start date for data (YYYY-MM-DD) | 2023-01-01 | 2023-06-01 |
| `--end-date` | End date for data (YYYY-MM-DD) | 2023-03-31 | 2023-08-31 |
| `--bbox` | Bounding box: min_lon min_lat max_lon max_lat | -120 35 -115 40 | -130 42 -120 49 |
| `--output-dir` | Output directory for results | output | ./results |
| `--test-size` | Test set size ratio (0.1-0.5) | 0.2 | 0.3 |
| `--random-state` | Random seed for reproducibility | 42 | 123 |

### Geographic Regions Examples

```bash
# California
python run_model.py --bbox -124 32 -114 42

# Texas  
python run_model.py --bbox -106 25 -93 36

# Florida
python run_model.py --bbox -87 24 -80 31

# Pacific Northwest
python run_model.py --bbox -130 42 -116 49

# Great Lakes
python run_model.py --bbox -92 41 -76 49
```

## 🔍 Understanding the Output

### Performance Metrics Interpretation

- **MAE (Mean Absolute Error)**: Average prediction error in °C
  - `< 2°C`: Excellent performance
  - `2-5°C`: Good performance  
  - `> 5°C`: Needs improvement

- **RMSE (Root Mean Square Error)**: Penalizes larger errors
  - Should be close to MAE for consistent errors
  - Much larger than MAE indicates outlier issues

- **R² Score**: Explained variance (0-1, higher is better)
  - `> 0.8`: Excellent model
  - `0.6-0.8`: Good model
  - `< 0.6`: Needs improvement

### Data Sources

The model uses real-world data from:
- **Primary**: NOAA Climate Data Online API
- **Fallback**: Realistic synthetic data based on meteorological principles
- **Variables**: Temperature, precipitation, geographic coordinates, time features

### Machine Learning Models

1. **Linear Regression**: 
   - Baseline model for interpretability
   - Captures linear temperature relationships

2. **Random Forest**:
   - Handles non-linear patterns
   - Captures feature interactions
   - Provides feature importance

3. **Ensemble**:
   - Combines both models using simple averaging
   - More robust than individual models

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure virtual environment is activated
   source venv/bin/activate  # macOS/Linux
   venv\Scripts\activate     # Windows
   
   # Reinstall dependencies
   pip install -r requirements/base.txt
   ```

2. **Data Download Issues**
   - The script uses a fallback synthetic data generator if NOAA API is unavailable
   - No action needed - the model will still work with realistic data

3. **Memory Issues**
   ```bash
   # Reduce data size with shorter time period
   python run_model.py --start-date 2023-01-01 --end-date 2023-01-31
   ```

4. **Permission Errors**
   ```bash
   # Ensure write permissions in current directory
   chmod 755 .
   
   # Or specify different output directory
   python run_model.py --output-dir ~/Documents/climate_results
   ```

### Getting Help

```bash
# View all available options
python run_model.py --help

# Check if environment is set up correctly
python -c "import pandas, sklearn, matplotlib; print('All dependencies available')"
```

## 📚 Next Steps

After running the local model:

1. **Analyze Results**: Review the HTML report and plots
2. **Experiment**: Try different regions and time periods
3. **Understand Features**: Look at feature importance plots
4. **Compare Models**: Examine individual model performance
5. **Deploy**: Use the cloud infrastructure for production deployment

## 🔗 Related Documentation

- [Main README](../README.md) - Full project overview
- [Cloud Setup Guide](docs/cloud-setup.md) - AWS deployment
- [Architecture Documentation](docs/architecture.md) - System design
- [API Documentation](src/api/) - REST API usage

## 📧 Support

For questions or issues:
1. Check the troubleshooting section above
2. Review the HTML report for model insights
3. Examine the console output for error details
4. Open an issue in the GitHub repository