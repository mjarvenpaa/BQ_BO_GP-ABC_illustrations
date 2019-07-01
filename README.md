### Simple 1d illustrations of posterior uncertainty in various Gaussian process emulator-based algorithms such as

1. **Bayesian quadrature (BQ)**
  * See `BQ_demo` 
  * For some mathematical details see e.g. the [Bayesian Monte Carlo paper](http://mlg.eng.cam.ac.uk/zoubin/papers/RasGha03.pdf) and the [Bayes-Hermite quadrature paper](https://www.sciencedirect.com/science/article/pii/037837589190002V)

2. **Bayesian optimisation (BO)**
  * See `BO_uncertainty_demo` which computes, by simulating sample paths from the GP numerically, the **posterior of the max and argmax of the unknown black-box function** that is modelled with GP as typically in BO. 
  * The above quantities are used in Information-theoretic BO methods such as [Entropy search](http://jmlr.csail.mit.edu/papers/volume13/hennig12a/hennig12a.pdf) and [Max-value entropy search](http://proceedings.mlr.press/v70/wang17e.html).

3. **GP surrogate based ABC/LFI inference (GP-ABC/GP-LFI)**
  * In [Approximate Bayesian Computation](https://en.wikipedia.org/wiki/Approximate_Bayesian_computation) (**ABC**) setting the discrepancy function between simulated and observed data has been modelled with GP (for details see e.g. [this paper](http://jmlr.org/papers/v17/15-017.html) and [this one](https://projecteuclid.org/euclid.ba/1537258134)) which is here called **GP-ABC**.
  * As a side note, the above approach is sometimes called **BOLFI** although, as discussed in [this paper](https://projecteuclid.org/euclid.ba/1537258134) in detail, it is better to not treat GP-ABC as BO problem but as an instance of more general **optimal design/active learning** problem because the goal of ABC inference is to obtain posterior approximation and not to optimise any objective function as in (standard) BO.
  * Additionally, the (approximate) log-likelihood function (see e.g. [here](http://jmlr.org/papers/v17/15-017.html) and [here](https://arxiv.org/abs/1905.01252)) has been modelled with GP which is here called **GP-LFI**, where LFI refers to **likelihood-free inference**.
  * See `GPABC_uncertainty_demo` and `GPLFI_uncertainty_demo` which demonstrate the GP-induced posterior over the ABC/LFI likelihood function, ABC/LFI approximate posterior, its expectation and evidence.
  * Note that approximate BQ methods have been developed, in the latter case where the log-likelihood is modelled with GP surrogate, to compute model evidence [here](https://papers.nips.cc/paper/4657-active-learning-of-model-evidence-using-bayesian-quadrature) and [here](http://proceedings.mlr.press/v89/chai19a.html). In this case, the evidence depends on the log-likelihood, which follows GP, in a **nonlinear** fashion and Gaussian approximations have been proposed in the two papers above. However, as the `GPLFI_uncertainty_demo` shows, the **the posterior of evidence is non-Gaussian**. In fact, even log(evidence) can have long right tail so that the **Gaussian approximation of the evidence can be very poor!**

Note: This repo does not contain any practical algorithms but only visualisations and related computations which I made to better understand
* how related problems to BQ, BO and the recent GP-accelerated ABC/LFI methods relate to each other,
* how does the GP-induced posterior uncertainty look like (e.g. in the case 1., if the likelihood is modelled with GP, then the evidence is always normally distributed but this is not the case for the quantity-of-interest in BO. Further, in ABC case, the quantity-of-interest, i.e. the approximate posterior of the simulation model, is typically infinite-dimensional!),
* how much uncertainty is left after some fixed amount evaluations.

Note: this is partially work-in-progress. 
