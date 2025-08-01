# kernel/equations.py (Versão Corrigida)

import numpy as np
from numpy.linalg import norm

# As funções agora recebem os parâmetros do config em vez de importá-lo.

def calculate_propensional_asymmetry(rho: np.ndarray) -> float:
    if rho.size == 0:
        return 0.0
    return np.var(rho)

def calculate_reflexive_energy(E: np.ndarray) -> float:
    if E.size == 0:
        return 0.0
    return norm(E)

def calculate_archetypal_energy(resonances: list[float], E_psi: float, e_psi_min_threshold: float, delta_k_decay: float) -> float:
    if E_psi < e_psi_min_threshold:
        valid_resonances = [r for r in resonances if r > delta_k_decay]
        if not valid_resonances:
            return 0.0
        return max(valid_resonances)
    return 0.0

def calculate_omega(A_rho: float, E_psi: float, omega_basal: float, lambda_diss: float, mu_diss: float, nu_diss: float) -> float:
    reactive_term = (1 - (mu_diss * E_psi + nu_diss * A_rho))
    omega = omega_basal + lambda_diss * reactive_term
    return max(0, omega)

def calculate_C1(E_psi: float, S_psi: float, time_step: int, k_sigmoid: float, e_threshold: float, c_max: float, beta_rhythm: float, nu_e_freq: float, phi_phase: float) -> float:
    psi_total = E_psi + S_psi
    exponent = -k_sigmoid * (psi_total - e_threshold)
    sigmoid_term = c_max / (1 + np.exp(exponent))
    rhythmic_term = (beta_rhythm * E_psi) * np.sin(nu_e_freq * time_step + phi_phase)
    c1 = sigmoid_term + rhythmic_term
    return max(0, c1)