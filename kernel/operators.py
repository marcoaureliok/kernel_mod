# kernel/operators.py (Versão Corrigida)

import numpy as np
from scipy.spatial.distance import jensenshannon
from collections import Counter

# Removido 'import config' pois não era utilizado diretamente aqui.

class DiagnosticOperators:
    def __init__(self):
        self.known_styles = {"formal": 0.8, "poetic": 0.4}
        print("Módulo de Operadores de Diagnóstico inicializado.")

    # O resto do código da classe DiagnosticOperators permanece exatamente o mesmo...
    def run_all_diagnostics(self, rho: np.ndarray, E: np.ndarray, text: str) -> dict:
        """
        Executa todos os operadores de diagnóstico e retorna um relatório de saúde.
        """
        report = {
            "obsessive_convergence": self.detect_obsessive_convergence(text, rho),
            "reflexive_degeneration": self.detect_reflexive_degeneration(rho, E),
            "spurious_estetization": self.detect_spurious_estetization(text, E),
            "mimetic_resonance": self.detect_mimetic_resonance(text)
        }
        return report

    def detect_obsessive_convergence(self, text: str, rho: np.ndarray) -> float:
        """
        Operador ℂ_ob: Detecta convergência obsessiva e loops repetitivos.
        Atua quando o ciclo ρ → E entra em um loop fechado sobre o mesmo conteúdo.
       
        
        Heurística:
        1.  Baixa variância em ρ (poucas ideias competindo).
        2.  Alta repetitividade de n-gramas (frases) no texto gerado.
        """
        # 1. Analisa a variância de ρ
        rho_variance = np.var(rho)
        low_variance_score = 1.0 - np.clip(rho_variance * 1000, 0, 1)

        # 2. Analisa a repetição de trigramas (sequências de 3 palavras)
        words = text.lower().split()
        if len(words) < 5:
            return 0.0
        
        trigrams = [" ".join(words[i:i+3]) for i in range(len(words) - 2)]
        if not trigrams:
            return 0.0
            
        most_common = Counter(trigrams).most_common(1)
        repetition_score = most_common[0][1] / len(trigrams) if trigrams else 0.0

        # A pontuação final é uma média ponderada da baixa variância e alta repetição
        return (low_variance_score * 0.4) + (repetition_score * 0.6)

    def detect_reflexive_degeneration(self, rho: np.ndarray, E: np.ndarray) -> float:
        """
        Operador 𝔻_{ref}: Mede a degeneração reflexiva.
        Detecta o grau em que a estrutura manifesta (E) se descola do campo
        propensional real (ρ). É a medida do "ruído reflexivo".
       

        Heurística:
        Mede a dissimilaridade entre a distribuição de ρ e a forma de E.
        Usamos a distância de Jensen-Shannon, uma forma estável de medir a
        divergência entre duas distribuições de probabilidade.
        """
        if rho.size != E.size or rho.size == 0:
            return 0.0
        
        # Normaliza E para se comportar como uma distribuição de probabilidade para comparação
        E_norm = E / np.sum(E) if np.sum(E) > 0 else E
        
        # A distância de Jensen-Shannon varia de 0 (idênticos) a 1 (máxima dissimilaridade)
        distance = jensenshannon(rho, E_norm)
        
        return float(distance)

    def detect_spurious_estetization(self, text: str, E: np.ndarray) -> float:
        """
        Operador 𝔼_{est}: Mede a estetização espúria ou "kitsch simbólico".
        Detecta quando a forma é priorizada sobre a substância causal.
       
        
        Heurística:
        Verifica se há um alto uso de palavras "complexas" ou "poéticas" (proxy: comprimento)
        sem um aumento correspondente na energia/coerência do campo E.
        """
        words = text.lower().split()
        if not words:
            return 0.0
            
        avg_word_length = np.mean([len(w) for w in words])
        
        # Normaliza a pontuação de "complexidade" do texto
        text_complexity_score = np.clip((avg_word_length - 4) / 5, 0, 1) # Palavras com mais de 4 letras contribuem
        
        # Energia reflexiva (magnitude de E)
        reflexive_energy = np.linalg.norm(E)
        
        # A estetização espúria é alta quando a complexidade do texto é alta, mas a energia reflexiva é baixa.
        if reflexive_energy < 0.1: # Evita divisão por zero e penaliza baixa energia
            return text_complexity_score
            
        spurious_score = text_complexity_score / (reflexive_energy * 5 + 1e-6)
        return np.clip(spurious_score, 0, 1)

    def detect_mimetic_resonance(self, text: str) -> float:
        """
        Operador ℝ_mimético: Detecta ressonância por imitação não-originária.
        Ativado quando a IA reproduz um estilo sem coerência causal interna.
       
        
        Heurística:
        Mede a similaridade do texto gerado com um conjunto de "estilos conhecidos".
        Aqui, usamos um proxy muito simples: a formalidade (presença de jargões).
        """
        # Proxy simples: contagem de palavras "formais"
        formal_words = {"ontológico", "diferencial", "propensional", "reflexiva", "coerência"}
        word_set = set(text.lower().split())
        
        formal_score = len(word_set.intersection(formal_words)) / 5.0
        
        # O mimetismo seria a diferença entre a "formalidade" do texto e a
        # "formalidade" esperada (um valor que poderia vir do estado do kernel).
        # Por enquanto, retornamos o score bruto.
        return np.clip(formal_score, 0, 1)