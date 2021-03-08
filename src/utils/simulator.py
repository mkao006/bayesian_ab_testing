''' Simulate traffic for AB testing
'''
import torch
import pyro
import pyro.distributions as dist

class ProportionTrafficGenerator():
    def __init__(self,
                 N: torch.tensor,
                 p: torch.tensor,
                 traffic_size: torch.tensor,                 
                 variant_proportion: torch.tensor):
        self.N = N
        self.p = p        
        self.traffic_size = traffic_size
        self.variant_proportion = variant_proportion
        self.control_size = (self.traffic_size * self.variant_proportion).floor()
        self.variant_size = self.traffic_size - self.control_size
        self.split_traffic_size = torch.stack([self.control_size, self.variant_size])

    def generate_traffic(self, normalise: bool=False) -> torch.tensor:
        if normalise:
            return dist.Binomial(self.split_traffic_size, self.p).sample((self.N, ))/\
                self.split_traffic_size
        else:
            return dist.Binomial(self.split_traffic_size, self.p).sample((self.N, ))
