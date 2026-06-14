# Industrial Soft Sensor Using Just-In-Time Learning and PLS

## Overview

This project develops an adaptive industrial soft sensor for predicting product quality variables using process measurements. The work was motivated by the challenge of delayed laboratory measurements in industrial processes, where real-time quality information is often unavailable.

The project was developed in two stages:

1. Implementation and evaluation of multiple global regression models:

   * Linear Regression
   * Ridge Regression
   * Lasso Regression
   * Polynomial Regression
   * Polynomial + Ridge Regression
   * Polynomial + Lasso Regression

2. Development of a Just-In-Time (JIT) learning framework using Partial Least Squares (PLS) as the local prediction model.

The final solution combines:

* Rolling-window learning
* Mahalanobis distance-based sample selection
* Local PLS model construction
* Online database updates using laboratory feedback

This adaptive approach allows the model to respond to process drift and changing operating conditions more effectively than conventional global models.

---

## Key Features

* Industrial soft-sensor development framework
* Just-In-Time (JIT) learning
* Partial Least Squares (PLS) local modeling
* Mahalanobis distance-based neighborhood selection
* Rolling-window adaptation using recent process history
* Support for baseline global regression models
* Online model updating using laboratory feedback
* Designed for dynamic industrial processes

---

## Results

The performance of the proposed JIT-PLS soft sensor was evaluated using historical industrial process data.

### Model Comparison

| Model                 | R²         | RMSE       |
| --------------------- | ---------- | ---------- |
| Linear Regression     | 0.1258 | 0.1678 |
| Ridge Regression      | 0.1258 | 0.1678 |
| Lasso Regression      | 0.0217 | 0.1775 |
| Polynomial Regression | 0.0074 | 0.1788 |
| Polynomial + Ridge    | 0.0074 | 0.1788 |
| Polynomial + Lasso    | 0.0780 | 0.1723 |
| JIT-PLS (Degree: 1)   | 0.7921 | 0.0737 |
| JIT-PLS (Degree: 2)   | 0.7909 | 0.0739 |

## Methodology

### Stage 1: Global Regression Models

The project development began with the implementation and evaluation of several global regression models:

* Linear Regression
* Ridge Regression
* Lasso Regression
* Polynomial Regression
* Polynomial + Ridge Regression
* Polynomial + Lasso Regression

These models were trained using the available historical process data and served as baseline soft-sensor models.

While global models can achieve good predictive performance, they often struggle when process operating conditions change over time. Industrial processes are inherently dynamic, and a single global model may not remain optimal across all operating regions.

### Stage 2: Just-In-Time (JIT) PLS Soft Sensor

To improve adaptability under changing operating conditions, a Just-In-Time (JIT) learning framework was implemented using Partial Least Squares (PLS) as the local regression model.

Instead of relying on a single global model, a local PLS model is generated for every prediction request.

The workflow is:

1. Receive a new process sample.
2. Calculate Mahalanobis distance between the query sample and historical observations.
3. Select the most relevant neighboring samples.
4. Build a local PLS model using the selected neighborhood.
5. Predict the target quality variable.
6. Update the rolling-window database when the corresponding laboratory measurement becomes available.

This approach allows the soft sensor to adapt to process drift and changing operating conditions while maintaining prediction accuracy.

---

## Rolling Window Strategy

The model operates using a fixed-size rolling database containing the most recent 200 process observations.

For every new laboratory result:

* The new sample is added to the database.
* The oldest sample is removed.
* The rolling window size remains constant.

This ensures that model updates are based on the most relevant and recent process behavior while reducing the influence of outdated operating conditions.

---

## JIT-PLS Workflow

```text
Process Measurements
         │
         ▼
Rolling Window Database
(Recent 200 Samples)
         │
         ▼
Mahalanobis Distance
Calculation
         │
         ▼
Nearest Neighbor Selection
         │
         ▼
Local PLS Model Construction
         │
         ▼
Quality Prediction
         │
         ▼
Laboratory Result Available
         │
         ▼
Rolling Window Update
```

---

## Hyperparameters

The JIT-PLS framework provides several configurable parameters that influence model behavior and prediction performance.

Examples include:

* Rolling window size
* Number of nearest neighbors
* Number of PLS latent variables
* Mahalanobis distance configuration
* Neighborhood selection criteria

Refer to the implementation for complete parameter definitions and tuning options.

---

## Modeling Approach

The following models were evaluated during development:

| Model                 | Type           |
| --------------------- | -------------- |
| Linear Regression     | Global         |
| Ridge Regression      | Global         |
| Lasso Regression      | Global         |
| Polynomial Regression | Global         |
| Polynomial + Ridge    | Global         |
| Polynomial + Lasso    | Global         |
| JIT-PLS               | Local Adaptive |

The JIT-PLS approach was developed to address limitations of static global models under changing process conditions.

---

## Validation Strategy

Traditional validation approaches such as TimeSeriesSplit were not used because the objective of this project was to replicate actual industrial soft-sensor deployment.

The evaluation procedure follows real operational behavior:

1. Maintain a rolling database of recent process observations.
2. Predict the next quality value using currently available information.
3. Wait for laboratory analysis to become available.
4. Update the rolling database with the newly validated sample.
5. Continue prediction for subsequent observations.

This methodology closely represents real-world online soft-sensor operation where future laboratory measurements are unavailable at prediction time.

---

## Installation

```bash
git clone https://github.com/vinaydhara2002/Industrial-Soft-Sensor-JIT-PLS.git

cd Industrial-Soft-Sensor-JIT-PLS

pip install -r requirements.txt
```
---

## Dataset Availability

The industrial dataset used in this project is not publicly available due to confidentiality restrictions.

The repository focuses on the implementation, methodology, and adaptive soft-sensor framework. No proprietary process data has been included.
