---
title: Topic Index
shorttitle: Topics
layout: default
---

### Foundations - Math
- [Jensen’s Inequality: The line connecting two points on a parabola is above the parabola] (wiki/jensens.html)
- [Calculus: Stretching, Curvature, and Max/Mins] (wiki/Derivatives)


### Foundations - Probability
- [The Basics of Probability](wiki/probability_basics.html)
- [Probability Rules] (wiki/probability_rules.html)
- [Single-variable Probabilities: PMFs, Densities, and CDFs] (wiki/distributions.html)
- [Multivariable Probabilities: Joint, Marginal, and Conditional distributions](wiki/joint_marginal_conditional.html)
- [Independence] (wiki/independence.html)
- [Bayes Rule] (wiki/bayes_rule.html)
- [Expectations] (wiki/expectations.html)
- [Law of Large Numbers: Averages converge] (wiki/LLN.html)
- [Central Limit Theorem: The particular average you get follows a normal distribution] (wiki/CLT.html)
- Hoefding's Inequality: Even finite sums aren't too wrong
- TODO: roll back this CLT to just talk about sums of RVs; talk about utility in "in practice"(?)

### Foundations - Distributions
- Distributions Cheat Sheet
- [TODO: break this up](wiki/distributions.html)
- Gaussian Distribution
- Multi-Dimensional Gaussian
- Bernoulli Distribution
- Binomial Distribution
- Poisson Distribution
- Exponential Distribution
- Cauchy
- Uniform
- Beta
- Gamma


### Foundations - Data
- [The Myth of "The" Data] (wiki/the_data_myth.html)
- [Sampling Distributions: that number you crunched could have been different] (wiki/sampling_distribution.html)
- [Law of Large Numbers in practice] (wiki/LLN_in_practice.html)
- [Central Limit Theorem in practice] (wiki/CLT_in_practice.html)


### Foundations - Statistics
- [Statistical Models and Fitting] (wiki/what_is_a_model.html)
- [Fitting method: Method of Moments] (wiki/method_of_moments.html)
- [Fitting method: Maximum Likelihood] (wiki/MLE.html)
- Identifiability: Sometimes there are two answers. Try to avoid that.
- p-values: How rare (what percentile) is the data under a given parameter setting?
- Confidence Intervals: Drawn so that if you re-collected the data and re-made the intervals 95% of the intervals would contain the true value, whatever it is
- [Terminology] (wiki/stats_terms.html)

### Foundations - Models
- [Linear regression (as least squares): Define a loss function and run with it] (wiki/lin_reg_ols)
- [Linear regression (as a probability model): Add distributional assumptions to make stronger statements] (wiki/lin_reg_probability.html)
- [GLMs: Use distributional assumptions appropriate to your context] (wiki/glm.html)
- [Logistic Regression: a particular GLM] (wiki/log_reg.html)


### Foundations - Simulation
- [Basic Monte Carlo] (wiki/basicmontercarlo.html)
- [Law of Large Numbers applied to simulation] (wiki/LLN_applied_to_simulation.html)
- [Central Limit Theorem applied to simulation] (CLT_applied_to_simulation.html)


### Kinds of Models
- [Types of Models] (wiki/types_of_models.html)
- [Generative vs Discriminative: Model everything, or only what you need?] (wiki/generativemodels.html)
- [Mixture Models: Distributions which roll a die to pick a distribution] (wiki/mixture_models.html)
- [Supervised vs Unsupervised Learning] (wiki/typesoflearning.html)


### Machine Learning - Fitting
- [Learning (aka Fitting) a Model] (wiki/noiseless_learning.html)
- [Learning with Noise: Bias and Variance: Inflexible models can better ignore noise] (wiki/noisylearning.html)
- [Test set: to measure model performance on fresh data] (wiki/test_set.html)
- [Overfitting: Overly-Flexible models will do poorly out-of-sample] (wiki/overfitting.html)
- [Regularization: Combat overfitting by making the model pay for flexibility] (wiki/regularization.html)


### Machine Learning - Cross Validation
- [Validation and Cross Validation] (wiki/validation.html)
- [Cross Validation: Why 5-fold CV is almost always the right choice] (wiki/k_fold.html)


### Bootstrapping
- [Bootstrapping: Simulating new samples of data] (wiki/bootstrap.html)
- Bootstrapping to get a sampling distribution, confidence interval, or p value


### Information Theory
- [Entropy: Surprise, and Difficulty of Guessing] (wiki/Entropy.html)
- [Optional: Thermodynamic Entropy] (wiki/thermo_entropy.html)
- [KL divergence: How much do the data surprise our model?] (wiki/KL_divergence.html)
- [Divergence and Deviance: How different are two distributions? How wrong is a model?] (wiki/Deviance.html)
- Maximum entropy distributions (wiki/maxent.html)


### Decision Theory
- [Classification Risk] (wiki/classificationrisk.html)
- [ERM] (wiki/erm.html)


### Model Comparison (Choosing one of several models)
- [Model Comparison using in-sample information criteria] (wiki/modelcompar.html)
- KL (again), AIC, BIC	
- [Model Comparison continued] (wiki/modelcompar2.html)


### Optimization:
- [Gradient Descent in one dimension] (wiki/optimcalc.html)
- [Gradient Descent and SGD] (wiki/gradientdescent.html)
- [Simulated Annealing] (wiki/simanneal.html)


### Sampling from a Distribution
- Samples vs PDFs: why we like having a sample
- [Inverse Transform for sampling] (wiki/inversetransform.html)
- [Rejection Sampling] (wiki/rejectionsampling.html)


### Variance Reduction
- [Importance Sampling: ] (wiki/importancesampling.html)
- [stratified Sampling] (wiki/stratification.html)


### MCMC - Basics
- Intro: Markov processes can and often do converge
- [Markov Chains and MCMC] (wiki/markov.html)


### MCMC - Convergence
- Effective Sample Size
- [Formal Tests for Convergence] (wiki/gewecke.html)
- Convergence Test: Geweke
- Convergence Test: Gelman-Rubin


### MCMC - Metropolis-Hastings
- Metroplis	 (wiki/metropolis.html)
- [Metropolis-Hastings] (wiki/metropolishastings.html)
- [Discrete MCMC] (wiki/discretemcmc.html)
- [Step size considerations] (wiki/convergenceandcoverage.html)


### MCMC - Gibbs Sampling
- [Intro to Gibbs Sampling] (wiki/introgibbs.html)
- [Gibbs from Metropolis-Hastings] (wiki/gibbsfromMH.html)
- [Gibbs with conditional a conjugate] (wiki/gibbsconj.html)
- Gibbs on a graphical model	
- [A gibbs sampler with lots of autocorrelation] (wiki/tetchgibbs.html)


### MCMC - Data Augmentation
- [Data Augmentation] (wiki/dataaug.html)
- [Slice Sampling] (wiki/slice.html)


### MCMC - HMC
- [The Idea of Hamiltonian Monte Carlo] (wiki/hmcidea.html)
- [Exploring Hamiltonian Monte Carlo] (wiki/hmcexplore.html)


### MCMC - Tuning
- Centering to help HMC
- [L, epsilon, and other tweaking] (wiki/hmctweaking.html)
- [Gelman Schools and Hierarchical Pathology (Funnels)] (wiki/gelmanschools.html)
- [Marginalizing over Discretes] (wiki/marginaloverdiscrete.html)


### Bayes - Basics
- Bayesian (wiki/bayes.html)
- Choosing Priors (wiki/priors.html)
- Poserior, Posterior Predictive (and credible intervals)
- Example: [Beta-Binomial and Globe Tossing] (wiki/globemode.html)
- Example: [Fitting a normal distribution to data] (wiki/normalmodel.html)


### Bayes - Regression
- Example: [Fitting a normal model in pymc3] (wiki/normalmodelwithpymc.html)
- [Bayesian Regression] (wiki/bayesianregression.html)
- [From the normal model to regression]	(wiki/pymcnormalreg.html)
- [Regression with custom priors] (wiki/reguninfprior.html)


### Bayes - Pooling & Hierarchical Models
- Pooling: 100 county-sized averages or one state-sized average?
- [Hierarchical Models] (wiki/hierarch.html)
- Fitting via Method of Moments
- Fitting via Full Bayes
- [Tumors in Rats] (wiki/tumorlab.html)


### Bayes - GLMs
- [Generalized Linear Models] :	(wiki/monksglm.html)
- [Drunk Monks: 0-inflated poisson regression]: (wiki/monksglm2.html)
- [Poisson Regression - tools on islands, part 1] (wiki/Islands1.html)
- [Poisson Regression - tools on islands part 2] (wiki/Islands2.html)

### Modeling Correlations between slopes and intercepts



### Mixture Models
- [Mixture Models, and types of learning] (wiki/typesoflearning.html)
- [Mixtures and MCMC] (wiki/mixtures_and_mcmc.html)


### Model Fitting
- [Maximum Likelihood] (wiki/MLE.html)
- Method of Moments
- [The EM algorithm] (wiki/EM.html)
- [EM with indices] (wiki/EM.html)
- [EM example] (wiki/EM.html)
- [Variational Inference] (wiki/VI.html)
- [ADVI] (wiki/advi.html)


### Gaussian Processes
- [The idea behind the GP] (wiki/GP1.html)
- [Gaussian Processes and 'Non-parametric' Bayes] (wiki/GP2.html)
- [Inference for GPs] (wiki/gp3.html)


### ???
- [Gelman Schools Theory] (wiki/gelmanschoolstheory.html)
- [Poisson-Gamma] (wiki/sufstatexch.html)


###Technologies
[SKlearn] (wiki/sklearn.html)


### Labs
- [Lab 7: Bayesian inference with PyMC3] (wiki/Lab7_bioassay.html)
- [Lab 8: A Brief Intro to Theano] (wiki/BrieIntroToTheano.html)
- [Lab 8: Data Augmentation and Slice Sampling] (wiki/LabSliceandDA.html)
- [Lab 10: Prosocial Chimps] (wiki/prosocialchimps.html)
- [Lab 12: Correlations] (wiki/corr.html)
- [Lab 12: Geographic Correlation and Oceanic Tools] (wiki/gpcorr.html)
- [Lab 12: Gaussian Mixture Model with ADVI] (wiki/gaussian-mixture-model-advi.html)
