# The Page Curve and Information Preservation in v7 Holographic Fisher Geometry

## 1. Introduction: The Black Hole Information Paradox

### 1.1 The Problem

Hawking's calculation (1975) showed that black holes emit thermal radiation at temperature:

```
T_H = ℏc³/(8πGMk_B)
```

This radiation appears to be exactly thermal - carrying no information about what fell into the black hole. If the black hole completely evaporates, the information seems to be lost, violating quantum mechanical unitarity.

### 1.2 The Page Curve

Don Page (1993) argued that if black hole evaporation is unitary, the entanglement entropy of the radiation must follow a characteristic curve:

```
S_rad(t)
    │
    │        ╱╲
    │       ╱  ╲
    │      ╱    ╲
    │     ╱      ╲
    │    ╱        ╲
    │   ╱          ╲
    │  ╱            ╲
    └─────────────────── time
       ↑            ↑
    early        late
   (rising)    (falling)
         ↑
     Page time
```

**Key features:**
- **Early times**: S_rad increases as radiation becomes entangled with the black hole
- **Page time**: S_rad reaches maximum (~half the initial BH entropy)
- **Late times**: S_rad decreases as information transfers to radiation

The Page time occurs approximately when half the black hole has evaporated:
```
t_Page ~ (G_N M₀³)/(ℏc⁴)
```

---

## 2. Recent Breakthroughs: The Island Formula

### 2.1 The QES Prescription (2019-2020)

Penington, Almheiri, Engelhardt, Marolf, and Maxfield derived the Page curve using the **island formula**:

```
S(R) = min[ext(Area(∂I)/(4G_N) + S_bulk(R ∪ I))]
```

Where:
- **R** = radiation region (outside black hole)
- **I** = "island" (region inside black hole included in entropy calculation)
- **∂I** = boundary of the island (quantum extremal surface)
- **S_bulk** = bulk quantum field theory entropy

### 2.2 The Key Insight

At late times, the entropy is minimized by including an **island** inside the black hole:

- **Early times (no island)**: S(R) = S_bulk(R) ~ number of emitted quanta (grows)
- **Late times (with island)**: S(R) = Area(∂I)/(4G_N) + S_bulk(R ∪ I) (shrinks with BH)

The transition between these regimes produces the Page curve.

---

## 3. Connection to v7 Framework

### 3.1 The Master Equation

The v7 Holographic Fisher Geometry framework:

```
g_μν = ℓ_P²(G_μν^Fisher + γ₀E_μν)
```

Contains the conceptual ingredients for the Page curve:

| Component | Role in Page Curve |
|-----------|-------------------|
| G_μν^Fisher | Encodes state distinguishability of radiation |
| E_μν | Tracks BH-radiation entanglement |
| γ₀ | Couples entanglement to geometry |

### 3.2 Physical Interpretation

**Early times (rising S_rad):**
- E_μν dominates: BH-radiation entanglement grows
- Each Hawking quantum is maximally entangled with its partner behind the horizon
- The entanglement strain tensor captures this growing correlation

**Page time (maximum S_rad):**
- Transition point where E_μν contribution equals G_μν^Fisher contribution
- The quantum extremal surface jumps to include the island

**Late times (falling S_rad):**
- G_μν^Fisher of radiation dominates
- Information encoded in Fisher information metric transfers to radiation
- Entanglement between radiation subsystems becomes detectable

### 3.3 The Entanglement Strain Tensor

The v7 entanglement strain tensor:

```
E_μν = ∂_μS_ent · ∂_νS_ent - ½η_μν(∂S_ent)²
```

During evaporation:
- S_ent initially grows (more entanglement across horizon)
- ∂_μS_ent points radially inward (entanglement concentrated at horizon)
- E_μν peaks at the Page time
- After Page time, island formation causes E_μν to decrease

---

## 4. Current Implementation Status

### 4.1 What the Simulator Currently Tracks

| Quantity | Status | File |
|----------|--------|------|
| S_BH(t) - Black hole entropy | ✅ Implemented | `examples/black_hole.py` |
| M(t) - Mass evolution | ✅ Implemented | `examples/black_hole.py` |
| T(t) - Hawking temperature | ✅ Implemented | `examples/black_hole.py` |
| Radiation flux | ✅ Implemented | `examples/black_hole.py` |
| **S_rad(t) - Radiation entropy** | ❌ Not yet | Planned |
| **Page time detection** | ❌ Not yet | Planned |

### 4.2 What's Needed for Page Curve

1. **Radiation entropy tracking**: Compute S_rad = ∫(dM·c²)/(k_B·T) dt
2. **Page time detection**: Find t where dS_rad/dt = 0
3. **Total entropy verification**: Check S_BH + S_rad bounded by initial S_BH
4. **v7 connection**: Show E_μν peaks at Page time

---

## 5. Theoretical Requirements for Rigorous Derivation

### 5.1 Full Derivation Would Require

1. **Explicit Hilbert space decomposition**:
   ```
   H_total = H_BH ⊗ H_radiation
   ```

2. **Hawking pair dynamics**:
   ```
   |Ψ⟩ = Σ_n c_n |n⟩_BH ⊗ |n⟩_rad
   ```

3. **Partial trace for radiation density matrix**:
   ```
   ρ_rad = Tr_BH(|Ψ⟩⟨Ψ|)
   ```

4. **Von Neumann entropy**:
   ```
   S_rad = -Tr(ρ_rad log ρ_rad)
   ```

5. **Quantum extremal surface identification**:
   ```
   δS_gen/δγ = 0 where S_gen = Area(γ)/(4G_N) + S_bulk
   ```

### 5.2 Connection to Replica Wormholes

The island formula was derived using gravitational path integrals with replica wormholes. A complete v7 derivation would need to show that:

- The Fisher metric path integral reproduces replica wormhole contributions
- Island saddles emerge from entanglement strain extremization
- The QES condition follows from v7 master equation variation

---

## 6. Simplified Numerical Approach

### 6.1 Information Flow Model

For numerical demonstration, we use the information flow approximation:

**Radiation entropy increment:**
```
dS_rad = (dM · c²)/(k_B · T_Hawking)
```

This captures the essential physics:
- Each emitted quantum carries energy dE = dM·c²
- At temperature T, entropy per quantum ~ E/T
- Cumulative radiation entropy grows then falls

**Why this works:**
- Early times: High T → low entropy per quantum, but many quanta
- Late times: Low T → high entropy per quantum, but BH nearly gone
- Page time emerges naturally from this competition

### 6.2 Expected Results

For a black hole with initial mass M₀:

| Quantity | Expected Value |
|----------|----------------|
| Initial S_BH | πr_s²/(ℓ_P²) = 4πG²M₀²/(ℏc) |
| Page time | ~0.65 × t_evap (see note below) |
| Max S_rad | ~0.5 × S_BH(0) |
| Final S_rad | → 0 (information recovered) |

**Note on Page time**: The naive expectation is t_Page ~ 0.5 × t_evap (when half the entropy
has been radiated). However, with the cubic-root mass evolution M(t) = M₀(1-t/t_evap)^(1/3),
the Page time shifts to ~0.65 × t_evap. This is because:
- Entropy scales as S ∝ M², so S(t) = S₀(1-t/t_evap)^(2/3)
- Page time occurs when S_rad = S_BH, i.e., when (1-t/t_evap)^(2/3) = 0.5
- Solving: t_Page/t_evap = 1 - 0.5^(3/2) ≈ 0.65

---

## 7. Open Questions

### 7.1 Within v7 Framework

1. **Can the island be derived from v7?**
   - Does extremizing g_μν = ℓ_P²(G_μν^Fisher + γ₀E_μν) give QES condition?

2. **Role of Immirzi parameter γ₀**:
   - Does γ₀ = 0.274 affect the Page time?
   - Is there a γ₀ correction to the entropy formula?

3. **Fisher metric of radiation**:
   - How does G_μν^Fisher of the radiation field encode BH information?
   - Can we compute this explicitly?

### 7.2 General Open Questions

1. **Interior reconstruction**: How is the BH interior reconstructed from radiation?
2. **Firewalls**: Does the v7 framework avoid the firewall paradox?
3. **Scrambling time**: How does information spread before Page time?

---

## 8. Conclusion

The Page curve represents the definitive test of unitarity in black hole evaporation. The v7 Holographic Fisher Geometry framework has the conceptual ingredients to address this:

- **Entanglement strain tensor E_μν** tracks BH-radiation entanglement
- **Fisher information metric G_μν^Fisher** encodes quantum state distinguishability
- **The master equation** unifies geometry and information

A full derivation would be a significant research contribution. The current implementation plan provides a numerical demonstration that captures the essential physics while connecting to v7 concepts.

---

## References

1. Page, D.N. (1993). "Information in black hole radiation," *Phys. Rev. Lett.* **71**, 3743.

2. Hawking, S.W. (1975). "Particle creation by black holes," *Commun. Math. Phys.* **43**, 199.

3. Penington, G. (2020). "Entanglement wedge reconstruction and the information paradox," *JHEP* **09**, 002.

4. Almheiri, A., Engelhardt, N., Marolf, D., & Maxfield, H. (2019). "The entropy of bulk quantum fields and the entanglement wedge of an evaporating black hole," *JHEP* **12**, 063.

5. Almheiri, A., Mahajan, R., Maldacena, J., & Zhao, Y. (2020). "The Page curve of Hawking radiation from semiclassical geometry," *JHEP* **03**, 149.

---

*This document accompanies the v7 Holographic Fisher Geometry proposal.*
*See also: theory/quantum-gravity-proposal-v7.md*
