''' Simulate traffic for AB testing
'''
import torch
import pyro.distributions as dist


class ProportionTrafficGenerator():
    def __init__(self,
                 N: torch.tensor,
                 p: torch.tensor,
                 traffic_size: torch.tensor,
                 treatment_proportion: torch.tensor):
        self.N = N
        self.p = p
        self.traffic_size = traffic_size
        self.treatment_proportion = treatment_proportion
        self.control_size = (self.traffic_size * (1 - self.treatment_proportion)).floor()
        self.variant_size = self.traffic_size - self.control_size
        self.split_traffic_size = torch.stack([self.control_size, self.variant_size])

    def generate_traffic(self) -> torch.tensor:
        self.sample = dist.Binomial(self.split_traffic_size, self.p).sample((self.N, ))
        self.summarise_traffic()

    def summarise_traffic(self):
        self.x0 = self.sample[:, 0].sum()
        self.n0 = self.split_traffic_size[0] * self.N

        self.x1 = self.sample[:, 1].sum()
        self.n1 = self.split_traffic_size[1] * self.N
