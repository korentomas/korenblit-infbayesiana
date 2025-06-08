#https://github.com/BayesianModelingandComputationInPython/BookCode_Edition1/tree/main/notebooks

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
from scipy import stats
from scipy.stats import entropy
from scipy.optimize import minimize

with pm.Model() as model:
    # Specify the prior distribution of unknown parameter
    θ = pm.Beta("θ", alpha=1, beta=1)
    #
    # Specify the likelihood distribution and condition on the observed data
    y_obs = pm.Binomial("y_obs", n=1, p=θ, observed=Y)
    #
    # Sample from the posterior distribution
    idata = pm.sample(1000, return_inferencedata=True)


# Summary statistics
summary = az.summary(idata)
print(summary)

# Trace plot
az.plot_trace(idata)
plt.show()

# Posterior plot
az.plot_posterior(idata)
plt.show()


