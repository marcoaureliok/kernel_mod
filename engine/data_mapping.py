# engine/data_mapping.py (Versão Corrigida)

import numpy as np
from scipy.stats import entropy

# A função agora recebe o 'rho_space_dimension' e 'c_max' como argumentos.

def map_to_rho(ia_internal_states: dict, rho_space_dimension: int) -> np.ndarray:
    logits = ia_internal_states.get("logits", np.array([]))
    
    if logits.size != rho_space_dimension:
        # Garante que o vetor de logits tenha a dimensão esperada
        # Isso é um fallback, em um caso real poderia levantar um erro.
        padded_logits = np.zeros(rho_space_dimension)
        size = min(logits.size, rho_space_dimension)
        padded_logits[:size] = logits[:size]
        logits = padded_logits

    if logits.size > 0:
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)
    else:
        probabilities = np.array([])
        
    local_entropy = entropy(probabilities + 1e-9)
    rho_field = probabilities + (local_entropy * probabilities * 0.5)
    
    if np.sum(rho_field) > 0:
        return rho_field / np.sum(rho_field)
    return rho_field

def map_to_E(ia_generated_text: str, rho_space_dimension: int, c_max: float) -> np.ndarray:
    if not ia_generated_text:
        return np.zeros(rho_space_dimension)
        
    coherence_score = len(ia_generated_text.split()) / 50.0
    
    words = ia_generated_text.split()
    if not words:
        complexity_score = 0
    else:
        complexity_score = np.mean([len(w) for w in words]) / 10.0
    
    focus_point = int(coherence_score * rho_space_dimension) % rho_space_dimension
    E_field = np.zeros(rho_space_dimension)
    peak_height = complexity_score * c_max
    
    if peak_height > 0:
        indices = np.arange(rho_space_dimension)
        sigma = 5.0 
        E_field = peak_height * np.exp(-((indices - focus_point)**2 / (2 * sigma**2)))

    return E_field