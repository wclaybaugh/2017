---
title: Levels of Bayesian Analysis
shorttitle: levelsofbayes
noline: 1
layout: wiki
keywords: ['bayesian']
summary: "Multiple levels of data analysis go from Frequentist to more and more Bayesian, starting from MLE and going to full bayes via Type-2 MLE (Empirical Bayes). The outstanding observation about these levels is the increasing reliance on integration as we get more Bayesian."
---

Throughout this course, you have (and will) see multiple levels of analysis. These levels are summarized in this table, taken from Murphy's Machine Learning from a Probabilistic Perspective book:



| Method                 | Definition                               |
| ---------------------- | ---------------------------------------- |
| Maximum Likelihood     | $\hat{\theta} = argmax_{\theta} p(D \vert \theta)$ |
| MAP estimation         | $\hat{\theta} = argmax_{\theta} p(D \vert \theta)p(\theta \vert \eta)$ |
| ML-2 (Empirical Bayes) | $\hat{\eta} = argmax_{\eta} \int d\theta \,p(D \vert \theta)p(\theta \vert \eta) = argmax_{\eta}p(D \vert \eta)$ |
| MAP-2                  | $\hat{\eta} = argmax_{\eta} \int d\theta \,p(D \vert \theta)p(\theta \vert \eta)p(\eta) = argmax_{\eta}p(D \vert \eta)p(\eta)$ |
| Full Bayes             | $p(\theta, \eta \vert D) \propto p(D \vert \theta)p(\theta \vert \eta) p(\eta)$ |



### Maximum Likelihood

This is the first technique we learned and used. There is no notion of priors here, juat a likelihood $\cal{L}$ which we maximize (or equivalently minimize cost $-\ell = - log(\cal{L})$ .

### MAP

Once we decide to commit being bayesian, we construct a posterior. We can decide to mostly ignore the posterior and just use the parameter values at the mode of the posterior. This is the next level of analysis, where we run optimization on the functional form of the posterior. Essentially we are sampling from the likelihood (sampling distribution) at the mode of the posterior. This is reductive, but often much easier to find than the full posterior.


### Empirical Bayes (ML-2)

The third level is to compute the posterior predictive distribution analytically, and optimize that against the hyper-priors. That is, we do the integrals to marginalize out everything but the hyper-prior parameters, and get P(Dataset|Hyper-prior parameters). We then select the hyper-prior parameters that maximize this 'likelihood'. [Hence the alternative designation "maximum likelihood type 2"]

Since we're finding single values for the hyperparameters, we've effectively selected a prior distribution- indeed, the one which best matches the dataset- rather than specifying a prior from knowledge available before data collection. The prior ends up specifed so we get all the features of bayesian analysis. However, we're discarding any variability in the hyperparameters and risk overfitting to our data. The Full Bayes method (below) would average over the possible values of the hyper parameters instead of running forward with a particular setting.

This is a very popular method when using conjugate priors, as the posterior predictive is often tractable (gamma+poisson gives negative binomial, beta+binomial gives beta-binomial, normal+normal gives normal). 

Often, in place of selecting the hyper parameters via optimizing/MLE, we will find the hyper-parameters by matching moments: match the mean of the posterior predictive against the mean of data, and similarly for variance.

### MAP-2

Remember that there are hyper-priors on the hyper-parameters? MAP-2 simply multiplies these into the posterior-predictive before optimizing. This allows hyper-parameters to influence our inference. But, since the game in hierarchical modelling is to be relatively un-informative on the hyper-priors, this typically does not change things much

### Full Bayes

Finally, we can be fully Bayesian. We compute the entire posterior and thus posterior-predictive by sampling, and marginalize to find the distribution of any parameter we want, and sample-compute the expectations we desire.

One can think of  ascending this hierarchy as one in which you carry out more integrals and fewer derivatives, and thus go in a direction where overfitting is controlled better and better.
