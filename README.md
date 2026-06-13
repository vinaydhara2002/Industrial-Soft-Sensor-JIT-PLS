# Industrial Soft Sensor for Debutanizer Column Quality Prediction

## Overview

This project implements a rolling-window Just-In-Time (JIT) soft sensor for predicting C4 mole fraction in a debutanizer column using industrial process measurements.

The framework combines:

- Mahalanobis-distance neighborhood selection
- Local weighted learning
- Partial Least Squares (PLS) regression
- Polynomial feature expansion
- Rolling-window adaptation

## Methodology

1. Build rolling database window
2. Transform features using polynomial expansion
3. Compute Mahalanobis distances
4. Select nearest operating-condition samples
5. Train weighted local PLS model
6. Predict product quality

## Results

- R² ≈ 0.80
- Local adaptive modeling of changing operating regimes

## Technologies

- Python
- NumPy
- Pandas
- SciPy
- Scikit-learn
- Matplotlib

## Dataset

Industrial dataset not included due to confidentiality restrictions.