# Task Time-to-End (TTE) Prediction for Efficient Scheduling

This repository contains the implementation and analysis of various prediction methods for task scheduling in resource-constrained environments. The project compares machine learning approaches (LSTM) with simpler statistical methods (Linear Regression, Exponential Smoothing) to evaluate the trade-offs between prediction accuracy and computational efficiency. Note that code was mainly written alongside Perplexity, Claude, Cursor, and ChatGPT.

## Project Structure

```
tte-predictions/
├── Paper Figures/       # Generated visualizations for research paper
├── Metrics/            # Performance metrics and analysis results
├── Results/           # Experimental results and data
├── resource_metrics_and_visualizations.py  # Resource usage monitoring and visualization
├── tte_predictor.py   # Core prediction models implementation
├── paper_visualizations.py  # Visualization code for paper figures
├── integration_example.py   # Example of integrating predictors in applications
└── main_experiment.ipynb   # Main experimental notebook
```

## Features

- Multiple prediction models:
  - LSTM (Machine Learning)
  - Linear Regression (Statistical)
  - Exponential Smoothing (Statistical)
- Resource usage monitoring and analysis
- Comprehensive performance metrics
- Visualization tools for analysis
- Integration examples

## Data

This project uses real-world data from the 2024 SCANIA dataset via 
Creative Commons Attribution 4.0 International (CC BY 4.0) license: https://researchdata.se/sv/catalogue/dataset/2024-34/2
It can also be fitted with similar TTE/clustering data, like from the Google clusters dataset.

## Installation

```bash
# Clone the repository
git clone https://github.com/anika-lakhani/tte-predictions.git
cd tte-predictions

# Install required packages
pip install -r requirements.txt
```

## Usage

### Quick Start

```python
from tte_predictor import TTEPredictor

# Initialize predictor
predictor = TTEPredictor()

# Train models
predictor.train(training_data)

# Make predictions
predictions = predictor.predict(test_data)
```

### Running Experiments

1. Open `main_experiment.ipynb` in Jupyter Notebook
2. Follow the notebook cells to:
   - Load and preprocess data
   - Train different models
   - Compare performance metrics
   - Generate visualizations

### Generating Visualizations

```python
from paper_visualizations import generate_figures

# Generate all paper figures
generate_figures()
```

## Performance Metrics

The project evaluates models on several key metrics:
- Prediction accuracy (MAE, RMSE)
- Computational efficiency
- Memory usage
- Training time
- Inference latency

## Resource Requirements

Minimum system requirements:
- Python 3.8+
- 4GB RAM
- CPU: 2+ cores

For LSTM models:
- 8GB RAM recommended
- GPU support optional but recommended

## Acknowledgements

Lindgren, T. et al. (2024) “SCANIA Component X Dataset: A Real-World Multivariate Time Series Dataset for Predictive Maintenance.” Scania CV AB. Tillgänglig via: https://doi.org/10.5878/jvb5-d390Opens in a new tab.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{tte_predictions2024,
  title={Determining Scenarios for Low-Fidelity Time-to-Event (TTE) Scheduling Prediction},
  author={Anika Lakhani},
  year={2025}
}
```
