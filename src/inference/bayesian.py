''' Bayesian AB testing

https://www.chrisstucchio.com/blog/2014/bayesian_ab_decision_rule.html

https://www.chrisstucchio.com/pubs/slides/gilt_bayesian_ab_2015/slides.html#1

https://www.evanmiller.org/bayesian-ab-testing.html

'''


from abc import ABC, abstractmethod
from typing import Callable
from functools import partial

import numpy as np
import pandas as pd
import torch
from torch import tensor
import pyro
import pyro.distributions as dist
from pyro.infer import (
    NUTS,
    MCMC
)

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')


class BayesianTester(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def model_generator(self) -> Callable:
        pass

    @abstractmethod
    def run(self) -> None:
        pass

    def loss(self, a, b) -> int:
        return np.maximum(a - b, 0)


class BayesianBinaryTester(BayesianTester):
    def __init__(self,
                 outcome: tensor,
                 traffic_size: tensor,
                 warmup_steps: int = 100,
                 num_samples: int = 1000) -> None:
        pyro.clear_param_store()
        self.outcome = outcome
        self.traffic_size = traffic_size
        self.model = self.model_generator()
        self.kernel = NUTS(self.model)
        self.mcmc = MCMC(self.kernel, warmup_steps=warmup_steps, num_samples=num_samples)

    def model_generator(self) -> Callable:
        def _model_(self):
            control_prior = pyro.sample('control_p', dist.Beta(1, 1))
            treatment_prior = pyro.sample('treatment_p', dist.Beta(1, 1))
            return pyro.sample('obs',
                               dist.Binomial(
                                   self.traffic_size,
                                   torch.stack([control_prior, treatment_prior])
                               ), obs=self.outcome)

        return partial(_model_, self)

    def run(self, n_samples=3000) -> None:
        self.mcmc.run()
        self.posterior_samples = self.mcmc.get_samples(n_samples)

    def expected_loss_switch(self) -> float:
        return torch.mean(self.loss(self.posterior_samples['treatment_p'],
                                    self.posterior_samples['control_p']))

    def expected_loss_stay(self) -> float:
        return torch.mean(self.loss(self.posterior_samples['control_p'],
                                    self.posterior_samples['treatment_p']))

    def improvement_probability(self) -> float:
        return torch.mean((self.loss(self.posterior_samples['treatment_p'],
                                     self.posterior_samples['control_p']) > 0).float())

    def summary(self) -> None:
        cost_of_switching = self.expected_loss_switch()
        cost_of_stay = self.expected_loss_stay()
        prob_improve = self.improvement_probability()
        print('Potential cost of not switching: {:.2%}'.format(cost_of_switching))
        print('Cost of wrong switch: {:.2%}'.format(cost_of_stay))
        print('Probability of Treatment is better: {:.2%}'.format(prob_improve))

    def plot_joint_posterior(self) -> None:
        posterior_df = pd.DataFrame(self.posterior_samples)
        g = sns.jointplot(x='control_p',
                          y='treatment_p',
                          data=posterior_df,
                          kind='kde',
                          fill=True,
                          levels=10)
        minimum = min([i.min() for i in self.posterior_samples.values()])
        maximum = max([i.max() for i in self.posterior_samples.values()])
        g.ax_marg_x.set_xlim(minimum, maximum)
        g.ax_marg_y.set_ylim(minimum, maximum)
        x0, x1 = g.ax_joint.get_xlim()
        y0, y1 = g.ax_joint.get_ylim()
        lims = [max(x0, y0), min(x1, y1)]
        g.ax_joint.plot(lims, lims, ':k')
        plt.show()

    def plot_posterior(self) -> None:
        posterior_melted_df = pd.melt(pd.DataFrame(self.posterior_samples))
        sns.kdeplot(data=posterior_melted_df, hue='variable', x='value')
        plt.show()


class BayesianCountTester(BayesianTester):
    def __init__(self, model):
        pass
