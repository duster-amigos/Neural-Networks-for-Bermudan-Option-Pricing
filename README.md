# Neural Networks for Bermudan Option Pricing

## Overview
This project implements neural network-based methods for pricing Bermudan options, following the approach described in "Neural network regression for Bermudan option pricing" by B. Lapeyre & J. Lelong (2021). The code simulates multi-dimensional Black-Scholes models, applies the Longstaff-Schwartz algorithm, and compares neural network regression with polynomial regression for option pricing.

---

## Theory

### 1. Black-Scholes Model for Multi-dimensional Assets
The price dynamics of a multi-dimensional asset S_t = (S_t^(1), ..., S_t^(d)) under the risk-neutral measure Q are given by:

    dS_t^(i) = r * S_t^(i) * dt + sigma_i * S_t^(i) * dW_t^(i)

- r: risk-free rate
- sigma_i: volatility of asset i
- dW_t^(i): Brownian motion (possibly correlated)

The solution is:

    S_t^(i) = S_0^(i) * exp((r - 0.5 * sigma_i^2) * t + sigma_i * W_t^(i))

#### Correlation
The Brownian motions can be correlated with correlation matrix Gamma. Cholesky decomposition is used to simulate correlated paths:

    W_t = L * B_t    where    Gamma = L * L^T

#### Brownian Motion Simulation
Brownian motion is a Gaussian process, characterized by its mean and covariance functions:
- Mean: m(t) = 0
- Covariance: K(s, t) = min(s, t)

To simulate a Gaussian vector with covariance Sigma:
1. Cholesky decomposition: Sigma = L * L^T
2. Generate standard normal vector G
3. Compute X = mu + L * G

#### Log-normality
For each dimension i, at time t, S_t^(i) follows a log-normal distribution:

    log(S_t^(i)) ~ N(log(S_0^(i)) + (r - 0.5 * sigma_i^2) * t, sigma_i^2 * t)

#### Statistical Properties
- Theoretical mean:    E_Q[S_t^(i)] = S_0^(i) * exp(r * t)
- Theoretical variance: Var_Q(S_t^(i)) = (S_0^(i))^2 * exp(2 * r * t) * (exp(sigma_i^2 * t) - 1)

#### Empirical Correlation
Empirical correlation matrix from simulated log-returns:

    R_t^(i, j) = log(S_{t+1}^{(i, j)}) - log(S_t^{(i, j)})

Covariance and correlation matrices are computed and compared to the theoretical correlation matrix used in simulation.

---

### 2. Bermudan Option Pricing
Bermudan options can be exercised at discrete times. The Longstaff-Schwartz algorithm is used for pricing, with regression (polynomial or neural network) to estimate continuation values.

#### Longstaff-Schwartz Algorithm Steps
1. Simulate asset price paths.
2. At each exercise date, regress the continuation value on state variables.
3. Decide whether to exercise or continue at each step.
4. Discount and average payoffs to estimate the option price.

#### Neural Network Regression
Feedforward neural networks (FFNN) are used as flexible regressors for the continuation value, compared to traditional polynomial regression.

---

### 3. Option Types and Payoff Functions
- **Put Option:**

      P = max(K - S_T, 0)

- **Geometric Basket Put Option:**

      P = max(K - (Product_{i=1}^d S_T^(i))^(1/d), 0)

- **Basket Put Option:**

      P = max(K - Sum_{i=1}^d w_i * S_T^(i), 0)

- **Max Call Option:**

      P = max(max_{i=1}^d S_T^(i) - K, 0)

---

### 4. Statistical Verification
- **Mean and Variance:** Empirical means and variances are compared to theoretical values, with confidence intervals.
- **Log-normality:** Normality tests (Kolmogorov–Smirnov, D’Agostino–Pearson) are used to verify the log-normal distribution of simulated prices.
- **Correlation:** Empirical correlation matrices are compared to the theoretical correlation matrix.

---

## Requirements
- Python 3.7+
- numpy
- pandas
- matplotlib
- networkx
- scipy
- tensorflow
- scikit-learn
- tqdm
- tqdm_joblib
- joblib

## References
- B. Lapeyre & J. Lelong (2021), "Neural network regression for Bermudan option pricing"
- Longstaff, F. A., & Schwartz, E. S. (2001). Valuing American options by simulation: a simple least-squares approach. The Review of Financial Studies, 14(1), 113-147.
- Black, F., & Scholes, M. (1973). The pricing of options and corporate liabilities. Journal of Political Economy, 81(3), 637-654.
