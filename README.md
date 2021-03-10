# Bayesian AB Testing

The code implements the Bayesian A/B testing framework in Pyro described in this [post](https://www.chrisstucchio.com/blog/2014/bayesian_ab_decision_rule.html) by Chris Stucchio.

There are numerous advantage of the framework such as the following but not limited to:

* Providing the probability of treatment better than control.
* Improved sensitivity and thus able to detect smaller changes.
* Quantification of the cost if a 'false positive' is made.
* Test can be stopped as soon the decision rule has been reached instead of waiting for a fixed amount of time.
* Takes into account of the gain in the test.

Resources:

[Bayesian A/B Testing at VWO by Chris Stucchio](https://www.chrisstucchio.com/pubs/slides/gilt_bayesian_ab_2015/slides.html#1)
[Formulas for Bayesian A/B Testing](https://www.evanmiller.org/bayesian-ab-testing.html)
[IS Bayesian A/B Testing Immune to Peeking? Not Exactly](http://varianceexplained.org/r/bayesian-ab-testing/)