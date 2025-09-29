# Running Climate Prediction Model Locally

This guide explains how to run the climate temperature prediction model on your local machine to generate predictions, visualizations, and comprehensive reports using ESA Climate Data for the Butzbach, Germany region. 

**🚀 GPU Support**: The model now supports GPU acceleration for faster training on CUDA-enabled systems, including RunPod containers.

## 🌍 About This Model

This project uses **ESA (European Space Agency) Climate Data** to predict temperatures in **Butzbach, Hessen, Germany** - a representative Central European location. The model demonstrates climate prediction capabilities using:

- **Data Source**: ESA Climate Data Portal and Copernicus Climate Data Store
- **Target Region**: Butzbach, Germany (50.4333°N, 8.6667°E) 
- **Climate Type**: Central European continental climate
- **Prediction**: Daily temperature variations with seasonal patterns
- **GPU Acceleration**: XGBoost and LightGBM with CUDA support when available

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- 2GB available disk space
- Internet connection (for downloading climate data)
- **Optional**: NVIDIA GPU with CUDA support for acceleration

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

# Install base dependencies
pip install -r requirements/base.txt
```

### Step 2: (Optional) GPU Setup

**For RunPod Users:**
```bash
# RunPod containers typically come with CUDA pre-installed
# Install GPU-enabled PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install GPU requirements
pip install -r requirements/gpu.txt
```

**For Local GPU Setup:**
```bash
# Ensure NVIDIA drivers and CUDA toolkit are installed
# Install GPU-enabled PyTorch (adjust CUDA version as needed)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install additional GPU requirements
pip install -r requirements/gpu.txt
```

### Step 3: Run the Model

```bash
# Basic run with default parameters (Butzbach, Germany region, 3 months of data)
python run_model.py

# The script will:
# 1. Detect GPU availability and configure acceleration
# 2. Download real climate data from ESA services  
# 3. Train machine learning models (with GPU acceleration if available)
# 4. Create visualization plots
# 5. Generate an HTML report
```

### Step 4: View Results

```bash
# Open the comprehensive HTML report
open output/climate_prediction_report.html  # macOS
start output/climate_prediction_report.html  # Windows
xdg-open output/climate_prediction_report.html  # Linux
```

## 💻 Hardware Performance

### CPU-Only Mode
- **Training Time**: ~30-60 seconds for 3 months of data
- **Models Used**: Linear Regression, Random Forest, CPU XGBoost, CPU LightGBM
- **Memory Usage**: ~500MB RAM

### GPU-Accelerated Mode  
- **Training Time**: ~10-30 seconds for 3 months of data (2-3x faster)
- **Models Used**: Linear Regression, Random Forest, GPU XGBoost, GPU LightGBM
- **Memory Usage**: ~1GB GPU memory + 500MB RAM
- **Requirements**: NVIDIA GPU with CUDA 11.8+ support

## 📊 Output Explanation

After running the model, you'll get:

### 📄 HTML Report (`output/climate_prediction_report.html`)
A comprehensive report including:
- **Executive Summary**: High-level results overview with GPU status
- **Performance Metrics**: MAE, RMSE, R² scores with explanations
- **Hardware Information**: GPU availability and model acceleration details
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
   - Comparison of different algorithms (including GPU vs CPU models)
   - Shows performance and training time differences
   - Demonstrates the value of ensemble approach and GPU acceleration

5. **`temperature_distribution.png`**
   - Histogram and box plot comparing actual vs predicted distributions
   - Ensures model captures the data distribution properly
   - Identifies any distribution shift issues

## ⚙️ Advanced Usage

### Custom Parameters

```bash
# Different geographic region (Frankfurt area)
python run_model.py --bbox 8.3 50.0 8.9 50.3

# Different time period (6 months)
python run_model.py --start-date 2023-01-01 --end-date 2023-06-30

# Custom output directory
python run_model.py --output-dir ./my_results

# Different train/test split
python run_model.py --test-size 0.3

# Complete custom run for entire Hessen region
python run_model.py \
    --start-date 2023-01-01 \
    --end-date 2023-12-31 \
    --bbox 8.0 49.8 10.2 51.8 \
    --output-dir ./hessen_results \
    --test-size 0.2
```

### GPU-Specific Options

The model automatically detects and uses GPU when available. To check GPU status:

```bash
# Check GPU availability
python -c "from src.models.gpu_utils import gpu_manager; print(gpu_manager.get_gpu_info())"

# Force CPU-only mode (if needed for testing)
CUDA_VISIBLE_DEVICES="" python run_model.py
```

### Parameter Descriptions

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--start-date` | Start date for data (YYYY-MM-DD) | 2023-01-01 | 2023-06-01 |
| `--end-date` | End date for data (YYYY-MM-DD) | 2023-03-31 | 2023-08-31 |
| `--bbox` | Bounding box: min_lon min_lat max_lon max_lat | 8.5 50.3 8.8 50.6 | 8.0 49.8 10.2 51.8 |
| `--output-dir` | Output directory for results | output | ./results |
| `--test-size` | Test set size ratio (0.1-0.5) | 0.2 | 0.3 |
| `--random-state` | Random seed for reproducibility | 42 | 123 |

### German Regions Examples

```bash
# Butzbach area (default)
python run_model.py --bbox 8.5 50.3 8.8 50.6

# Frankfurt metropolitan area
python run_model.py --bbox 8.3 50.0 8.9 50.3

# Entire Hessen state
python run_model.py --bbox 8.0 49.8 10.2 51.8

# Rhein-Main region
python run_model.py --bbox 8.1 49.9 8.8 50.3

# Kassel region (Northern Hessen)
python run_model.py --bbox 9.2 51.2 9.6 51.5
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
- **Primary**: ESA Climate Data Portal (https://climate.esa.int)
- **Secondary**: Copernicus Climate Data Store (CDS)
- **Fallback**: Realistic European climate data based on meteorological principles
- **Variables**: Temperature, precipitation, geographic coordinates, time features
- **Region Focus**: Central European continental climate patterns (Germany)

### Machine Learning Models

1. **Linear Regression**: 
   - Baseline model for interpretability
   - Captures linear temperature relationships

2. **Random Forest**:
   - Handles non-linear patterns
   - Captures feature interactions
   - Provides feature importance

3. **XGBoost** (CPU/GPU):
   - Gradient boosting for complex patterns
   - GPU acceleration when available
   - High performance on structured data

4. **LightGBM** (CPU/GPU):
   - Fast gradient boosting
   - Memory efficient
   - GPU support for speed

5. **Ensemble**:
   - Weighted combination of all models
   - GPU models get higher weight when available
   - Reduces overfitting and improves robustness

### GPU vs CPU Performance

| Aspect | CPU Mode | GPU Mode |
|--------|----------|----------|
| **Training Time** | 30-60 seconds | 10-30 seconds |
| **Models Used** | 4 (Linear, RF, XGB, LGB) | 4 (Linear, RF, XGB-GPU, LGB-GPU) |
| **Memory Usage** | ~500MB RAM | ~1GB VRAM + 500MB RAM |
| **Accuracy** | Baseline performance | Same or slightly better |
| **Scalability** | Limited by CPU cores | Better for larger datasets |
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
   - The script uses a fallback European climate data generator if ESA APIs are unavailable
   - No action needed - the model will still work with realistic German climate patterns

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