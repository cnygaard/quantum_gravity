import numpy as np
from scipy.constants import G, c, k, m_p

class StellarCore:
    def __init__(self, mass_solar, radius_solar, stellar_type):
        self.M = mass_solar * 1.989e30  # Solar mass in kg
        self.R = radius_solar * 6.957e8  # Solar radius in m
        self.type = stellar_type
        
    def central_density(self):
        """Calculate central density based on stellar type"""
        if self.type == 'main_sequence':
            return 1.5e5 * (self.M/1.989e30)**(2) * (self.R/6.957e8)**(-3)
        elif self.type == 'neutron_star':
            return 1e17
        elif self.type == 'white_dwarf':
            return 1e6 * (self.M/1.989e30)
        elif self.type == 'red_giant':
            return 1e4 * (self.M/1.989e30)
