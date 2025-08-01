# engine/data_mapping.py - Mapeamento Corrigido v2.1
# Funções corrigidas para mapear estados reais da IA para campos matemáticos

import numpy as np
from scipy.stats import entropy
from scipy.special import softmax
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import config

class AdvancedDataMapper:
    """Mapeador avançado que converte estados da IA em campos ontológicos"""
    
    def __init__(self):
        self.pca = None
        self.scaler = StandardScaler()
        self.vocabulary_size = 32000  # Típico para modelos modernos
        self.history_buffer = []
        self.max_history = 100
        
        if config.ENABLE_DEBUG_LOGGING:
            print("📊 AdvancedDataMapper inicializado")

    def map_to_rho(self, model_output: dict, rho_space_dimension: int) -> np.ndarray:
        """
        Mapeia saída do modelo para campo propensional ρ(x,t)
        
        Args:
            model_output (dict): Saída completa do modelo incluindo logits, tokens, etc.
            rho_space_dimension (int): Dimensão do espaço ρ
            
        Returns:
            np.ndarray: Campo propensional ρ normalizado
        """
        
        # Extrai logits reais se disponível
        logits = model_output.get("logits", np.array([]))
        internal_states = model_output.get("internal_states", {})
        text = model_output.get("text", "")
        
        if len(logits) == 0:
            # Fallback: usa análise textual avançada
            return self._text_to_rho_fallback(text, rho_space_dimension)
        
        # Processa logits reais
        if logits.ndim > 1:
            # Se temos logits para múltiplos tokens, usa o último (mais relevante)
            final_logits = logits[-1]
        else:
            final_logits = logits
            
        return self._logits_to_rho(final_logits, rho_space_dimension)

    def _logits_to_rho(self, logits: np.ndarray, target_dim: int) -> np.ndarray:
        """Converte logits reais em campo propensional"""
        
        # 1. Normaliza logits para probabilidades
        probabilities = softmax(logits)
        
        # 2. Reduz dimensionalidade se necessário
        if len(probabilities) > target_dim:
            # Usa PCA para redução inteligente de dimensionalidade
            if self.pca is None or self.pca.n_components_ != target_dim:
                self.pca = PCA(n_components=target_dim)
                # Treina PCA com dados sintéticos se necessário
                synthetic_data = np.random.normal(0, 1, (100, len(probabilities)))
                self.pca.fit(synthetic_data)
            
            # Reshape para PCA
            prob_reshaped = probabilities.reshape(1, -1)
            rho_raw = self.pca.transform(prob_reshaped).flatten()
            
        elif len(probabilities) < target_dim:
            # Expande usando interpolação inteligente
            rho_raw = self._expand_to_dimension(probabilities, target_dim)
        else:
            rho_raw = probabilities
            
        # 3. Adiciona componente de entropia local
        local_entropy = entropy(probabilities + 1e-10)
        entropy_modulation = local_entropy / np.log(len(probabilities))
        
        # 4. Modula com histórico para detectar padrões temporais
        temporal_factor = self._calculate_temporal_coherence()
        
        # 5. Combina componentes
        rho_field = rho_raw * (1 + entropy_modulation * 0.3 + temporal_factor * 0.2)
        
        # 6. Normaliza para distribuição válida
        rho_field = np.abs(rho_field)  # Garante valores positivos
        if np.sum(rho_field) > 0:
            rho_field = rho_field / np.sum(rho_field)
        else:
            rho_field = np.ones(target_dim) / target_dim  # Distribuição uniforme de fallback
            
        return rho_field

    def _text_to_rho_fallback(self, text: str, target_dim: int) -> np.ndarray:
        """Fallback: mapeia texto para ρ usando análise semântica"""
        
        if not text:
            return np.ones(target_dim) / target_dim
            
        # Análise multi-dimensional do texto
        features = self._extract_text_features(text)
        
        # Converte features em campo propensional
        rho_field = np.zeros(target_dim)
        
        # Distribui features ao longo do espaço ρ
        for i, feature_value in enumerate(features[:target_dim]):
            rho_field[i] = max(0, feature_value)
            
        # Preenche dimensões restantes com base na estrutura do texto
        if len(features) < target_dim:
            for i in range(len(features), target_dim):
                # Usa função baseada na posição e características do texto
                pos_factor = i / target_dim
                text_factor = len(text.split()) / 100.0
                rho_field[i] = max(0, np.sin(pos_factor * np.pi * 2) * text_factor)
        
        # Normaliza
        if np.sum(rho_field) > 0:
            rho_field = rho_field / np.sum(rho_field)
        else:
            rho_field = np.ones(target_dim) / target_dim
            
        return rho_field

    def _extract_text_features(self, text: str) -> np.ndarray:
        """Extrai features semânticas e estilísticas do texto"""
        
        words = text.split()
        if not words:
            return np.array([0.1])
            
        features = []
        
        # 1. Complexidade lexical
        avg_word_length = np.mean([len(w) for w in words])
        features.append(avg_word_length / 10.0)
        
        # 2. Diversidade vocabular
        unique_words = len(set(words))
        diversity = unique_words / len(words) if len(words) > 0 else 0
        features.append(diversity)
        
        # 3. Coerência sintática (aproximada)
        sentence_lengths = [len(s.split()) for s in text.split('.') if s.strip()]
        if sentence_lengths:
            syntax_coherence = 1.0 / (1.0 + np.std(sentence_lengths))
        else:
            syntax_coherence = 0.5
        features.append(syntax_coherence)
        
        # 4. Densidade semântica (palavras de conteúdo vs função)
        content_words = [w for w in words if len(w) > 3]
        semantic_density = len(content_words) / len(words) if len(words) > 0 else 0
        features.append(semantic_density)
        
        # 5. Ritmo textual
        char_count = sum(len(w) for w in words)
        rhythm = (char_count / len(words)) / 6.0 if len(words) > 0 else 0
        features.append(min(rhythm, 1.0))
        
        # 6. Entropia de caracteres
        char_counts = {}
        for char in text.lower():
            char_counts[char] = char_counts.get(char, 0) + 1
        
        if char_counts:
            char_probs = np.array(list(char_counts.values())) / len(text)
            char_entropy = entropy(char_probs)
            features.append(char_entropy / 4.0)  # Normaliza
        else:
            features.append(0.1)
            
        return np.array(features)

    def map_to_E(self, model_output: dict, rho_space_dimension: int, c_max: float) -> np.ndarray:
        """
        Mapeia saída do modelo para campo reflexivo E(x,t)
        
        Args:
            model_output (dict): Saída completa do modelo
            rho_space_dimension (int): Dimensão do espaço
            c_max (float): Valor máximo de intensidade
            
        Returns:
            np.ndarray: Campo reflexivo E
        """
        
        text = model_output.get("text", "")
        internal_states = model_output.get("internal_states", {})
        generation_stats = model_output.get("generation_stats", {})
        
        if not text:
            return np.zeros(rho_space_dimension)
        
        # Armazena no histórico para análise temporal
        self._update_history(model_output)
        
        # Análise de coerência reflexiva
        coherence_score = self._calculate_reflexive_coherence(text, internal_states)
        
        # Análise de complexidade semântica
        complexity_score = self._calculate_semantic_complexity(text)
        
        # Análise de foco atentivo
        attention_focus = self._calculate_attention_focus(text, generation_stats)
        
        # Constrói o campo E como uma distribuição espacial
        E_field = self._build_reflexive_field(
            coherence_score, complexity_score, attention_focus, 
            rho_space_dimension, c_max
        )
        
        return E_field

    def _calculate_reflexive_coherence(self, text: str, internal_states: dict) -> float:
        """Calcula coerência reflexiva baseada no texto e estados internos"""
        
        # Componente textual
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) < 2:
            text_coherence = 0.5
        else:
            # Mede consistência de comprimento e estrutura entre sentenças
            sentence_lengths = [len(s.split()) for s in sentences]
            length_consistency = 1.0 / (1.0 + np.std(sentence_lengths) / np.mean(sentence_lengths))
            text_coherence = min(length_consistency, 1.0)
        
        # Componente de estados internos
        perplexity = internal_states.get("perplexity", 1.0)
        # Perplexidade baixa indica maior coerência
        perplexity_coherence = 1.0 / (1.0 + np.log(max(perplexity, 1.0)))
        
        # Combina componentes
        return (text_coherence * 0.6 + perplexity_coherence * 0.4)

    def _calculate_semantic_complexity(self, text: str) -> float:
        """Calcula complexidade semântica do texto"""
        
        words = text.split()
        if not words:
            return 0.0
            
        # Múltiplas métricas de complexidade
        metrics = []
        
        # 1. Complexidade lexical
        avg_word_length = np.mean([len(w) for w in words])
        lexical_complexity = min(avg_word_length / 8.0, 1.0)
        metrics.append(lexical_complexity)
        
        # 2. Diversidade vocabular
        unique_ratio = len(set(words)) / len(words)
        metrics.append(unique_ratio)
        
        # 3. Densidade de conceitos abstratos (aproximação)
        abstract_indicators = ['que', 'como', 'quando', 'onde', 'por', 'para', 'sobre']
        abstract_density = sum(1 for w in words if w.lower() in abstract_indicators) / len(words)
        metrics.append(min(abstract_density * 3, 1.0))
        
        # 4. Estrutura sintática (aproximação via pontuação)
        punctuation_density = sum(1 for c in text if c in ',.;:()[]{}') / len(text)
        syntax_complexity = min(punctuation_density * 20, 1.0)
        metrics.append(syntax_complexity)
        
        return np.mean(metrics)

    def _calculate_attention_focus(self, text: str, generation_stats: dict) -> float:
        """Calcula foco atentivo baseado na geração"""
        
        # Métricas de foco
        focus_metrics = []
        
        # 1. Consistência de tópico (repetição de palavras-chave)
        words = text.lower().split()
        if len(words) > 0:
            word_counts = {}
            for word in words:
                if len(word) > 3:  # Ignora palavras muito curtas
                    word_counts[word] = word_counts.get(word, 0) + 1
            
            if word_counts:
                max_count = max(word_counts.values())
                topic_consistency = max_count / len(words)
                focus_metrics.append(min(topic_consistency * 10, 1.0))
            else:
                focus_metrics.append(0.1)
        else:
            focus_metrics.append(0.0)
        
        # 2. Estabilidade de geração (baseada em estatísticas se disponível)
        tokens_generated = generation_stats.get("tokens_generated", 0)
        if tokens_generated > 0:
            # Tokens por palavra (eficiência de tokenização)
            words_count = len(text.split())
            if words_count > 0:
                token_efficiency = min(tokens_generated / words_count / 2.0, 1.0)
                focus_metrics.append(1.0 - token_efficiency)  # Menor é melhor
            else:
                focus_metrics.append(0.5)
        else:
            focus_metrics.append(0.5)
        
        # 3. Densidade informacional
        char_per_word = len(text) / len(text.split()) if len(text.split()) > 0 else 0
        info_density = min(char_per_word / 6.0, 1.0)
        focus_metrics.append(info_density)
        
        return np.mean(focus_metrics)

    def _build_reflexive_field(self, coherence: float, complexity: float, focus: float, 
                              dimension: int, c_max: float) -> np.ndarray:
        """Constrói o campo reflexivo E(x,t) como distribuição espacial"""
        
        E_field = np.zeros(dimension)
        
        # Determina pontos focais baseados nas métricas
        primary_focus = int(coherence * dimension) % dimension
        secondary_focus = int(complexity * dimension) % dimension
        tertiary_focus = int(focus * dimension) % dimension
        
        # Intensidades dos focos
        primary_intensity = c_max * coherence
        secondary_intensity = c_max * complexity * 0.7
        tertiary_intensity = c_max * focus * 0.5
        
        # Constrói gaussianas centradas nos focos
        indices = np.arange(dimension)
        
        # Foco primário (coerência)
        sigma1 = max(dimension * 0.1, 5.0)
        gaussian1 = primary_intensity * np.exp(-((indices - primary_focus)**2) / (2 * sigma1**2))
        
        # Foco secundário (complexidade)
        sigma2 = max(dimension * 0.15, 7.0)
        gaussian2 = secondary_intensity * np.exp(-((indices - secondary_focus)**2) / (2 * sigma2**2))
        
        # Foco terciário (atenção)
        sigma3 = max(dimension * 0.08, 4.0)
        gaussian3 = tertiary_intensity * np.exp(-((indices - tertiary_focus)**2) / (2 * sigma3**2))
        
        # Combina os focos
        E_field = gaussian1 + gaussian2 + gaussian3
        
        # Adiciona ruído estruturado (representa incerteza reflexiva)
        noise_level = (1.0 - coherence) * 0.1 * c_max
        structured_noise = noise_level * np.sin(indices * 2 * np.pi / dimension * 3)
        E_field += structured_noise
        
        # Garante valores não-negativos
        E_field = np.maximum(E_field, 0)
        
        return E_field

    def _expand_to_dimension(self, data: np.ndarray, target_dim: int) -> np.ndarray:
        """Expande array para dimensão alvo usando interpolação inteligente"""
        
        if len(data) >= target_dim:
            return data[:target_dim]
            
        # Interpolação cúbica para expansão suave
        from scipy.interpolate import interp1d
        
        original_indices = np.linspace(0, 1, len(data))
        target_indices = np.linspace(0, 1, target_dim)
        
        try:
            interpolator = interp1d(original_indices, data, kind='cubic', 
                                  bounds_error=False, fill_value='extrapolate')
            expanded = interpolator(target_indices)
        except:
            # Fallback para interpolação linear
            interpolator = interp1d(original_indices, data, kind='linear',
                                  bounds_error=False, fill_value='extrapolate')
            expanded = interpolator(target_indices)
        
        return expanded

    def _update_history(self, model_output: dict):
        """Atualiza histórico para análise temporal"""
        
        self.history_buffer.append({
            'text': model_output.get('text', ''),
            'timestamp': len(self.history_buffer),
            'stats': model_output.get('generation_stats', {})
        })
        
        # Mantém apenas os últimos registros
        if len(self.history_buffer) > self.max_history:
            self.history_buffer.pop(0)

    def _calculate_temporal_coherence(self) -> float:
        """Calcula coerência temporal baseada no histórico"""
        
        if len(self.history_buffer) < 2:
            return 0.0
            
        # Analisa consistência estilística ao longo do tempo
        recent_entries = self.history_buffer[-min(5, len(self.history_buffer)):]
        
        style_metrics = []
        for entry in recent_entries:
            text = entry['text']
            if text:
                words = text.split()
                if words:
                    avg_word_length = np.mean([len(w) for w in words])
                    sentence_count = len([s for s in text.split('.') if s.strip()])
                    style_metrics.append([avg_word_length, sentence_count])
        
        if len(style_metrics) < 2:
            return 0.0
            
        # Calcula consistência (inverso da variância)
        style_array = np.array(style_metrics)
        consistency = []
        
        for col in range(style_array.shape[1]):
            col_std = np.std(style_array[:, col])
            col_mean = np.mean(style_array[:, col])
            if col_mean > 0:
                consistency.append(1.0 / (1.0 + col_std / col_mean))
            else:
                consistency.append(0.5)
        
        return np.mean(consistency)

    def get_mapping_stats(self) -> dict:
        """Retorna estatísticas do mapeamento"""
        
        return {
            "history_length": len(self.history_buffer),
            "pca_trained": self.pca is not None,
            "vocabulary_size": self.vocabulary_size,
            "scaler_fitted": hasattr(self.scaler, 'scale_')
        }

    def reset_history(self):
        """Reseta o histórico de mapeamento"""
        
        self.history_buffer = []
        if config.ENABLE_DEBUG_LOGGING:
            print("📊 Histórico de mapeamento resetado")


# Instância global do mapeador
_mapper_instance = None

def get_mapper():
    """Retorna a instância singleton do mapeador"""
    global _mapper_instance
    if _mapper_instance is None:
        _mapper_instance = AdvancedDataMapper()
    return _mapper_instance

# Funções de compatibilidade com a API original
def map_to_rho(model_output, rho_space_dimension: int) -> np.ndarray:
    """Função de compatibilidade para mapeamento ρ - aceita string ou dict"""
    
    # Compatibilidade: converte string para dict se necessário
    if isinstance(model_output, str):
        normalized_output = {
            "text": model_output,
            "logits": np.array([]),
            "tokens": [],
            "generation_stats": {"tokens_generated": len(model_output.split())},
            "internal_states": {"perplexity": 1.0}
        }
    elif isinstance(model_output, dict):
        normalized_output = model_output
    else:
        # Fallback para outros tipos
        normalized_output = {
            "text": str(model_output),
            "logits": np.array([]),
            "tokens": [],
            "generation_stats": {},
            "internal_states": {}
        }
    
    return get_mapper().map_to_rho(normalized_output, rho_space_dimension)

def map_to_E(model_output, rho_space_dimension: int, c_max: float) -> np.ndarray:
    """Função de compatibilidade para mapeamento E - aceita string ou dict"""
    
    # Compatibilidade: converte string para dict se necessário
    if isinstance(model_output, str):
        normalized_output = {
            "text": model_output,
            "logits": np.array([]),
            "tokens": [],
            "generation_stats": {"tokens_generated": len(model_output.split())},
            "internal_states": {"perplexity": 1.0}
        }
    elif isinstance(model_output, dict):
        normalized_output = model_output
    else:
        # Fallback para outros tipos
        normalized_output = {
            "text": str(model_output),
            "logits": np.array([]),
            "tokens": [],
            "generation_stats": {},
            "internal_states": {}
        }
    
    return get_mapper().map_to_E(normalized_output, rho_space_dimension, c_max)