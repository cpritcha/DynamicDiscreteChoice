# Roadmap

## Features

### General

- General discrete state and action spaces
- Model, data, algorithm separation of concerns

### Infinite Horizon NFXP Model

- Fit the model to a dataset
- Simulate data from a fitted model
- OPG confidence interval calculation on a fitted model

## Possible Additions

### Support General Transitions

- Add function based transitions

### Finited Horizon NFXP Model

- Add support for temporal transition

### Support Counterfactual Policy Queries

- Add methods to support comparing models

### Confidence Interval

- Parametric bootstrap
- Resampling bootstrap

### Model Fit Statistics

- Cross-validation

### Extend Random Utility Model

- [Hyperbolic discounting](http://www.sas.upenn.edu/~hfang/publication/wang/Fang-Wang-MS24403-R2.pdf)

### Implement Additional Estimators

- Nested Pseudo Likelihood
- Mathematical Program with Equilibrium Constraints (MPEC)
- Expectation Maximization (to account for across individual heterogeneity)

### Hypothesis Testing

- LR, Wald, Lagrange multiplier tests