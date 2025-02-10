def quantum_nfw_profile(self, r):
    """Q(r) = 1 + c(M)exp(-r/(rs×φ))√(Λ/(24×φ))"""
    rs = self.scale_radius
    mass_coupling = self.compute_mass_coupling()
    
    quantum_term = mass_coupling * np.exp(-r/(rs*self.phi))
    geometric_factor = np.sqrt(self.Lambda/(24*self.phi))
    
    return 1 + quantum_term * geometric_factor

