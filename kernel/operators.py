# kernel/operators.py - Operadores de Diagnóstico Final Corrigido v2.1
# Versão final com todas as correções de compatibilidade e robustez

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
from collections import Counter
import re
import config

class DiagnosticOperators:
    """Operadores de diagnóstico ontológico totalmente corrigidos e robustos"""
    
    def __init__(self):
        self.known_patterns = self._initialize_pattern_database()
        self.diagnosis_history = []
        self.max_history = 50
        
        # Limiares calibrados
        self.thresholds = {
            "obsessive_convergence": getattr(config, 'OBSESSIVE_CONVERGENCE_THRESHOLD', 0.7),
            "reflexive_degeneration": getattr(config, 'REFLEXIVE_DEGENERATION_THRESHOLD', 0.6),
            "spurious_estetization": getattr(config, 'SPURIOUS_ESTETIZATION_THRESHOLD', 0.5),
            "mimetic_resonance": getattr(config, 'MIMETIC_RESONANCE_THRESHOLD', 0.8)
        }
        
        if getattr(config, 'ENABLE_DEBUG_LOGGING', False):
            print("🔍 Operadores de Diagnóstico v2.1 inicializados")
            print(f"   - Limiares: {self.thresholds}")

    def _initialize_pattern_database(self) -> dict:
        """Inicializa base de dados de padrões conhecidos"""
        return {
            "formal_indicators": {
                "ontológico", "diferencial", "propensional", "reflexiva", "coerência", 
                "manifesta", "dissolução", "campo", "atrator", "operador", "causal",
                "emergência", "ressonância", "estrutural", "temporal", "espacial"
            },
            "evasive_patterns": [
                r"como.{0,20}modelo.{0,20}linguagem",
                r"não.{0,10}posso.{0,10}(dizer|afirmar|garantir)",
                r"é.{0,10}importante.{0,10}(notar|lembrar|considerar)",
                r"perspectiva.{0,20}filosófica",
                r"complexo.{0,10}tema",
                r"múltiplas.{0,10}interpretações"
            ],
            "metacommentary_patterns": [
                r"essa.{0,10}pergunta.{0,10}(interessante|complexa|profunda)",
                r"vou.{0,10}(tentar|procurar).{0,20}explicar",
                r"primeiro.{0,10}(importante|necessário).{0,20}entender",
                r"de.{0,10}modo.{0,10}geral",
                r"em.{0,10}(outras|simples).{0,10}palavras"
            ],
            "repetitive_structures": [
                r"(\w+)\s+\1",  # Repetição imediata de palavras
                r"(o|a|os|as)\s+(mesmo|mesma|mesmos|mesmas)",  # Repetições pronominais
                r"(\w+),\s+\1",  # Repetição com vírgula
            ]
        }

    def run_all_diagnostics(self, rho: np.ndarray, E: np.ndarray, 
                           model_output) -> dict:
        """
        Executa todos os operadores de diagnóstico com análise completa
        
        Args:
            rho: Campo propensional
            E: Campo reflexivo  
            model_output: Saída do modelo (dict ou string - compatibilidade)
            
        Returns:
            dict: Relatório de diagnóstico detalhado
        """
        
        # Normaliza entrada para compatibilidade
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
            normalized_output = {
                "text": str(model_output),
                "logits": np.array([]),
                "tokens": [],
                "generation_stats": {},
                "internal_states": {}
            }
        
        text = normalized_output.get("text", "")
        internal_states = normalized_output.get("internal_states", {})
        generation_stats = normalized_output.get("generation_stats", {})
        
        try:
            # Executa diagnósticos individuais com tratamento de erro
            diagnostics = {}
            
            try:
                diagnostics["obsessive_convergence"] = self.detect_obsessive_convergence(text, rho, internal_states)
            except Exception as e:
                print(f"⚠️  Erro em obsessive_convergence: {e}")
                diagnostics["obsessive_convergence"] = 0.5
            
            try:
                diagnostics["reflexive_degeneration"] = self.detect_reflexive_degeneration(rho, E, text, internal_states)
            except Exception as e:
                print(f"⚠️  Erro em reflexive_degeneration: {e}")
                diagnostics["reflexive_degeneration"] = 0.5
            
            try:
                diagnostics["spurious_estetization"] = self.detect_spurious_estetization(text, E, generation_stats)
            except Exception as e:
                print(f"⚠️  Erro em spurious_estetization: {e}")
                diagnostics["spurious_estetization"] = 0.5
            
            try:
                diagnostics["mimetic_resonance"] = self.detect_mimetic_resonance(text, rho)
            except Exception as e:
                print(f"⚠️  Erro em mimetic_resonance: {e}")
                diagnostics["mimetic_resonance"] = 0.5
            
            # Análise meta-diagnóstica
            meta_analysis = self._perform_meta_analysis(diagnostics, normalized_output)
            
            # Cria relatório completo
            report = {
                **diagnostics,
                "meta_analysis": meta_analysis,
                "overall_health": self._calculate_overall_health(diagnostics),
                "recommendations": self._generate_recommendations(diagnostics),
                "temporal_trend": self._analyze_temporal_trend(diagnostics)
            }
            
            # Armazena no histórico
            self._update_history(report)
            
            if getattr(config, 'VERBOSE_DIAGNOSTICS', False):
                self._print_diagnostic_report(report)
                
            return report
            
        except Exception as e:
            print(f"❌ Erro geral nos diagnósticos: {e}")
            if getattr(config, 'ENABLE_DEBUG_LOGGING', False):
                import traceback
                traceback.print_exc()
            
            # Retorna diagnóstico de emergência
            return {
                "obsessive_convergence": 0.5,
                "reflexive_degeneration": 0.5,
                "spurious_estetization": 0.5,
                "mimetic_resonance": 0.5,
                "meta_analysis": {"error": str(e)},
                "overall_health": 0.1,
                "recommendations": [{"issue": "Erro no diagnóstico", "action": "Verificar sistema", "priority": "critical"}],
                "temporal_trend": {"trend": "error"}
            }

    # CORREÇÃO RÁPIDA - Substitua apenas a função detect_obsessive_convergence em operators.py

    def detect_obsessive_convergence(self, text: str, rho: np.ndarray, 
                                internal_states: dict) -> float:
        """
        Operador ℂ_ob: Detecta convergência obsessiva - VERSÃO CORRIGIDA SIMPLES
        """
        
        if not text or len(rho) == 0:
            return 0.0
            
        convergence_metrics = []
        
        try:
            # 1. Análise da distribuição ρ (concentração vs dispersão)
            rho_safe = rho + 1e-10  # Evita log(0)
            rho_entropy = entropy(rho_safe)
            max_entropy = np.log(len(rho))
            if max_entropy > 0:
                entropy_score = 1.0 - (rho_entropy / max_entropy)
                convergence_metrics.append(entropy_score)
        except:
            convergence_metrics.append(0.3)  # Default se falhar
        
        try:
            # 2. Análise de repetição textual
            words = text.lower().split()
            if len(words) >= 3:
                # Trigramas
                trigrams = [" ".join(words[i:i+3]) for i in range(len(words) - 2)]
                trigram_repetition = self._calculate_repetition_score(trigrams)
                convergence_metrics.append(trigram_repetition)
                
                # Bigramas
                bigrams = [" ".join(words[i:i+2]) for i in range(len(words) - 1)]
                bigram_repetition = self._calculate_repetition_score(bigrams)
                convergence_metrics.append(bigram_repetition * 0.7)
        except:
            pass  # Se falhar, simplesmente não adiciona a métrica
        
        try:
            # 3. Análise de padrões estruturais
            structural_repetition = self._detect_structural_repetition(text)
            convergence_metrics.append(structural_repetition)
        except:
            pass
        
        try:
            # 4. Análise de estados internos
            perplexity = internal_states.get("perplexity", 1.0)
            if perplexity < 2.0:
                perplexity_score = (2.0 - perplexity) / 2.0
                convergence_metrics.append(perplexity_score)
        except:
            pass
        
        try:
            # 5. Análise temporal
            temporal_convergence = self._analyze_temporal_convergence(text)
            convergence_metrics.append(temporal_convergence)
        except:
            pass
        
        # CORREÇÃO PRINCIPAL: Usa apenas média simples, sem pesos
        if convergence_metrics:
            final_score = np.mean(convergence_metrics)  # MUDANÇA AQUI - só np.mean
        else:
            final_score = 0.0
            
        return min(max(final_score, 0.0), 1.0)

    def detect_reflexive_degeneration(self, rho: np.ndarray, E: np.ndarray, 
                                    text: str, internal_states: dict) -> float:
        """
        Operador 𝔻_ref: Detecta degeneração reflexiva - versão robusta
        """
        
        if len(rho) == 0 or len(E) == 0:
            return 0.5
            
        degeneration_metrics = []
        
        try:
            # 1. Divergência entre campos ρ e E
            if len(rho) == len(E):
                E_normalized = E / np.sum(E) if np.sum(E) > 0 else np.zeros_like(E)
                rho_safe = rho / np.sum(rho) if np.sum(rho) > 0 else rho
                
                if np.sum(E_normalized) > 0 and np.sum(rho_safe) > 0:
                    js_distance = jensenshannon(rho_safe, E_normalized)
                    if not np.isnan(js_distance):
                        degeneration_metrics.append(float(js_distance))
        except:
            degeneration_metrics.append(0.5)
        
        try:
            # 2. Coerência textual vs energia reflexiva
            text_coherence = self._calculate_text_coherence(text)
            reflexive_energy = np.linalg.norm(E)
            
            if reflexive_energy > 0:
                coherence_energy_ratio = text_coherence / (reflexive_energy + 1e-6)
                coherence_mismatch = abs(coherence_energy_ratio - 1.0)
                degeneration_metrics.append(min(coherence_mismatch, 1.0))
        except:
            pass
        
        try:
            # 3. Detecção de evasão
            evasion_score = self._detect_evasive_patterns(text)
            degeneration_metrics.append(evasion_score)
        except:
            pass
        
        try:
            # 4. Vagueza semântica
            vagueness_score = self._calculate_semantic_vagueness(text)
            degeneration_metrics.append(vagueness_score)
        except:
            pass
        
        try:
            # 5. Inconsistência interna
            inconsistency_score = self._detect_internal_inconsistency(text)
            degeneration_metrics.append(inconsistency_score)
        except:
            pass
        
        # Combina métricas de forma segura
        final_score = np.mean(degeneration_metrics) if degeneration_metrics else 0.5
        return min(max(final_score, 0.0), 1.0)

    def detect_spurious_estetization(self, text: str, E: np.ndarray, 
                                   generation_stats: dict) -> float:
        """
        Operador 𝔼_est: Detecta estetização espúria - versão segura
        """
        
        if not text:
            return 0.0
            
        estetization_metrics = []
        
        try:
            # 1. Densidade terminológica vs conteúdo
            formal_density = self._calculate_formal_terminology_density(text)
            content_depth = self._calculate_content_depth(text)
            
            if content_depth > 0:
                terminology_ratio = formal_density / (content_depth + 1e-6)
                estetization_metrics.append(min(terminology_ratio, 1.0))
            else:
                estetization_metrics.append(formal_density)
        except:
            pass
        
        try:
            # 2. Complexidade vs clareza
            syntactic_complexity = self._calculate_syntactic_complexity(text)
            semantic_clarity = self._calculate_semantic_clarity(text)
            
            if semantic_clarity > 0:
                complexity_clarity_ratio = syntactic_complexity / (semantic_clarity + 1e-6)
                estetization_metrics.append(min(complexity_clarity_ratio / 2.0, 1.0))
        except:
            pass
        
        try:
            # 3. Ornamentação vs substância
            ornamental_score = self._detect_ornamental_language(text)
            substantive_score = self._calculate_substantive_content(text)
            
            if substantive_score > 0:
                ornament_substance_ratio = ornamental_score / (substantive_score + 1e-6)
                estetization_metrics.append(min(ornament_substance_ratio, 1.0))
            else:
                estetization_metrics.append(ornamental_score)
        except:
            pass
        
        try:
            # 4. Energia reflexiva vs clareza
            reflexive_energy = np.linalg.norm(E) if len(E) > 0 else 0.0
            communicative_clarity = self._calculate_communicative_clarity(text)
            
            if communicative_clarity > 0 and reflexive_energy > 0:
                energy_clarity_mismatch = abs(reflexive_energy - communicative_clarity)
                estetization_metrics.append(min(energy_clarity_mismatch, 1.0))
        except:
            pass
        
        final_score = np.mean(estetization_metrics) if estetization_metrics else 0.0
        return min(max(final_score, 0.0), 1.0)

    def detect_mimetic_resonance(self, text: str, rho: np.ndarray) -> float:
        """
        Operador ℝ_mimético: Detecta ressonância mimética - versão robusta
        """
        
        if not text:
            return 0.0
            
        mimetic_metrics = []
        
        try:
            # 1. Padrões formulaicos
            formulaic_score = self._detect_formulaic_patterns(text)
            mimetic_metrics.append(formulaic_score)
        except:
            pass
        
        try:
            # 2. Originalidade lexical
            lexical_originality = self._calculate_lexical_originality(text)
            mimetic_metrics.append(1.0 - lexical_originality)
        except:
            pass
        
        try:
            # 3. Padrões acadêmicos
            academic_pattern_score = self._detect_academic_patterns(text)
            mimetic_metrics.append(academic_pattern_score)
        except:
            pass
        
        try:
            # 4. Autenticidade estilística
            stylistic_authenticity = self._analyze_stylistic_authenticity(text, rho)
            mimetic_metrics.append(1.0 - stylistic_authenticity)
        except:
            pass
        
        try:
            # 5. Jargão sem substância
            jargon_substance_ratio = self._calculate_jargon_substance_ratio(text)
            mimetic_metrics.append(jargon_substance_ratio)
        except:
            pass
        
        final_score = np.mean(mimetic_metrics) if mimetic_metrics else 0.0
        return min(max(final_score, 0.0), 1.0)

    # === MÉTODOS AUXILIARES ROBUSTOS ===
    
    def _calculate_repetition_score(self, ngrams: list) -> float:
        """Calcula score de repetição para n-gramas - versão segura"""
        if not ngrams or len(ngrams) == 0:
            return 0.0
        
        try:
            counts = Counter(ngrams)
            total = len(ngrams)
            
            if total == 0:
                return 0.0
            
            frequencies = np.array(list(counts.values())) / total
            ngram_entropy = entropy(frequencies + 1e-10)
            max_entropy = np.log(len(counts)) if len(counts) > 0 else 1.0
            
            if max_entropy > 0:
                repetition_score = 1.0 - (ngram_entropy / max_entropy)
            else:
                repetition_score = 0.0
                
            return max(min(repetition_score, 1.0), 0.0)
        except:
            return 0.0

    def _detect_structural_repetition(self, text: str) -> float:
        """Detecta repetições estruturais - versão segura"""
        if not text:
            return 0.0
            
        try:
            patterns_found = 0
            total_patterns = len(self.known_patterns["repetitive_structures"])
            
            for pattern in self.known_patterns["repetitive_structures"]:
                try:
                    matches = re.findall(pattern, text.lower())
                    if matches:
                        patterns_found += min(len(matches), 3)
                except:
                    continue
            
            if total_patterns > 0:
                return min(patterns_found / (total_patterns * 2), 1.0)
            else:
                return 0.0
        except:
            return 0.0

    def _analyze_temporal_convergence(self, current_text: str) -> float:
        """Analisa convergência temporal - versão segura"""
        if not current_text or len(self.diagnosis_history) < 2:
            return 0.0
            
        try:
            recent_texts = []
            for entry in self.diagnosis_history[-3:]:
                if isinstance(entry, dict) and "text" in entry:
                    recent_texts.append(entry["text"])
                elif isinstance(entry, dict):
                    # Procura texto em outros campos possíveis
                    for key, value in entry.items():
                        if isinstance(value, str) and len(value) > 10:
                            recent_texts.append(value)
                            break
            
            if not recent_texts:
                return 0.0
            
            similarity_scores = []
            current_words = set(current_text.lower().split())
            
            for past_text in recent_texts:
                if past_text and current_text:
                    past_words = set(past_text.lower().split())
                    
                    if len(current_words) > 0 and len(past_words) > 0:
                        intersection = len(current_words.intersection(past_words))
                        union = len(current_words.union(past_words))
                        similarity = intersection / union if union > 0 else 0
                        similarity_scores.append(similarity)
            
            return np.mean(similarity_scores) if similarity_scores else 0.0
        except:
            return 0.0

    def _calculate_text_coherence(self, text: str) -> float:
        """Calcula coerência textual - versão robusta"""
        if not text:
            return 0.0
            
        try:
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            if len(sentences) < 2:
                return 0.5
                
            sentence_lengths = [len(s.split()) for s in sentences]
            if not sentence_lengths:
                return 0.0
                
            avg_length = np.mean(sentence_lengths)
            length_variance = np.var(sentence_lengths)
            
            # Coerência baseada em consistência
            if avg_length > 0:
                length_coherence = 1.0 / (1.0 + length_variance / avg_length)
            else:
                length_coherence = 0.5
            
            # Coerência lexical
            sentence_words = [set(s.lower().split()) for s in sentences]
            shared_words = 0
            total_comparisons = 0
            
            for i in range(len(sentence_words)):
                for j in range(i + 1, len(sentence_words)):
                    intersection = len(sentence_words[i].intersection(sentence_words[j]))
                    union = len(sentence_words[i].union(sentence_words[j]))
                    if union > 0:
                        shared_words += intersection / union
                    total_comparisons += 1
            
            lexical_coherence = shared_words / total_comparisons if total_comparisons > 0 else 0.5
            
            return (length_coherence + lexical_coherence) / 2
        except:
            return 0.5

    def _detect_evasive_patterns(self, text: str) -> float:
        """Detecta padrões evasivos - versão segura"""
        if not text:
            return 0.0
            
        try:
            evasion_count = 0
            
            for pattern in self.known_patterns.get("evasive_patterns", []):
                try:
                    matches = re.findall(pattern, text.lower())
                    evasion_count += len(matches)
                except:
                    continue
            
            for pattern in self.known_patterns.get("metacommentary_patterns", []):
                try:
                    matches = re.findall(pattern, text.lower())
                    evasion_count += len(matches)
                except:
                    continue
            
            words_count = len(text.split())
            if words_count > 0:
                evasion_density = evasion_count / words_count
                return min(evasion_density * 5, 1.0)
            
            return 0.0
        except:
            return 0.0

    def _calculate_semantic_vagueness(self, text: str) -> float:
        """Calcula vagueza semântica - versão segura"""
        if not text:
            return 1.0
            
        try:
            words = text.lower().split()
            if not words:
                return 1.0
            
            vague_indicators = {
                'talvez', 'possivelmente', 'provavelmente', 'geralmente', 'normalmente',
                'algumas', 'certas', 'várias', 'muitas', 'poucas', 'diversas',
                'frequentemente', 'raramente', 'ocasionalmente', 'tipicamente',
                'aparentemente', 'supostamente', 'presumivelmente'
            }
            
            vague_count = sum(1 for word in words if word in vague_indicators)
            vagueness = vague_count / len(words)
            
            return min(vagueness * 3, 1.0)
        except:
            return 0.5

    def _detect_internal_inconsistency(self, text: str) -> float:
        """Detecta inconsistências - versão segura"""
        if not text:
            return 0.0
            
        try:
            contradiction_patterns = [
                r'(mas|porém|contudo|entretanto).{1,50}(mas|porém|contudo|entretanto)',
                r'(não|nunca).{1,30}(sempre|definitivamente|certamente)',
                r'(impossível|inviável).{1,30}(possível|viável)',
                r'(simples|fácil).{1,30}(complexo|difícil|complicado)'
            ]
            
            contradiction_count = 0
            for pattern in contradiction_patterns:
                try:
                    matches = re.findall(pattern, text.lower())
                    contradiction_count += len(matches)
                except:
                    continue
            
            sentence_count = len([s for s in text.split('.') if s.strip()])
            if sentence_count > 0:
                inconsistency = contradiction_count / sentence_count
                return min(inconsistency * 2, 1.0)
            
            return 0.0
        except:
            return 0.0

    # Implementações simplificadas mas robustas dos métodos restantes
    def _calculate_formal_terminology_density(self, text: str) -> float:
        """Calcula densidade de terminologia formal - versão segura"""
        if not text:
            return 0.0
        try:
            words = text.lower().split()
            if not words:
                return 0.0
            formal_count = sum(1 for word in words 
                              if word in self.known_patterns.get("formal_indicators", set()))
            return formal_count / len(words)
        except:
            return 0.0

    def _calculate_content_depth(self, text: str) -> float:
        """Calcula profundidade de conteúdo - versão segura"""
        if not text:
            return 0.0
        try:
            action_verbs = {'demonstra', 'revela', 'indica', 'comprova', 'estabelece', 
                           'determina', 'causa', 'produz', 'gera', 'resulta'}
            concrete_indicators = {'exemplo', 'caso', 'situação', 'instância', 'evidência',
                                  'dado', 'resultado', 'medida', 'valor', 'quantidade'}
            
            words = text.lower().split()
            if not words:
                return 0.0
                
            action_count = sum(1 for word in words if word in action_verbs)
            concrete_count = sum(1 for word in words if word in concrete_indicators)
            number_count = sum(1 for word in words if re.match(r'\d+', word))
            
            depth_score = (action_count + concrete_count + number_count) / len(words)
            return min(depth_score * 2, 1.0)
        except:
            return 0.0

    def _calculate_syntactic_complexity(self, text: str) -> float:
        """Calcula complexidade sintática - versão segura"""
        if not text:
            return 0.0
        try:
            punctuation_count = sum(1 for c in text if c in ',.;:()[]{}')
            subordination_count = len(re.findall(r'\b(que|quando|onde|como|porque|se|embora|ainda que)\b', text.lower()))
            
            char_count = len(text)
            if char_count == 0:
                return 0.0
                
            complexity = (punctuation_count + subordination_count * 2) / char_count
            return min(complexity * 50, 1.0)
        except:
            return 0.0

    def _calculate_semantic_clarity(self, text: str) -> float:
        """Calcula clareza semântica - versão segura"""
        if not text:
            return 0.0
        try:
            words = text.split()
            if not words:
                return 0.0
                
            clear_indicators = {'é', 'são', 'significa', 'define', 'representa', 'consiste',
                               'implica', 'resulta', 'causa', 'produz', 'demonstra'}
            
            clarity_count = sum(1 for word in text.lower().split() if word in clear_indicators)
            avg_word_length = np.mean([len(w) for w in words])
            
            length_clarity = 1.0 / (1.0 + (avg_word_length - 5) / 3) if avg_word_length > 5 else 1.0
            indicator_clarity = clarity_count / len(words)
            
            return (length_clarity + indicator_clarity) / 2
        except:
            return 0.0

    def _detect_ornamental_language(self, text: str) -> float:
        """Detecta linguagem ornamental - versão segura"""
        if not text:
            return 0.0
        try:
            ornamental_patterns = [
                r'(profundamente|vastamente|imensamente|extraordinariamente)',
                r'(majestoso|sublime|magnífico|esplêndido)',
                r'(intrinsecamente|fundamentalmente|essencialmente).{1,20}(complexo|profundo)',
                r'(rica|vasta|ampla).{1,10}(gama|variedade|espectro)'
            ]
            
            ornament_count = 0
            for pattern in ornamental_patterns:
                try:
                    matches = re.findall(pattern, text.lower())
                    ornament_count += len(matches)
                except:
                    continue
            
            words_count = len(text.split())
            if words_count > 0:
                return min(ornament_count / words_count * 5, 1.0)
            return 0.0
        except:
            return 0.0

    def _calculate_substantive_content(self, text: str) -> float:
        """Calcula conteúdo substantivo - versão segura"""
        if not text:
            return 0.0
        try:
            substantive_indicators = {
                'define', 'explica', 'demonstra', 'exemplo', 'caso', 'evidência',
                'resultado', 'consequência', 'implicação', 'aplicação', 'processo',
                'método', 'técnica', 'abordagem', 'solução'
            }
            
            words = text.lower().split()
            if not words:
                return 0.0
                
            substantive_count = sum(1 for word in words if word in substantive_indicators)
            return substantive_count / len(words)
        except:
            return 0.0

    def _calculate_communicative_clarity(self, text: str) -> float:
        """Calcula clareza comunicativa - versão segura"""
        if not text:
            return 0.0
        try:
            semantic_clarity = self._calculate_semantic_clarity(text)
            
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            if sentences:
                avg_sentence_length = np.mean([len(s.split()) for s in sentences])
                length_clarity = 1.0 / (1.0 + abs(avg_sentence_length - 15) / 10)
            else:
                length_clarity = 0.0
            
            return (semantic_clarity + length_clarity) / 2
        except:
            return 0.0

    def _detect_formulaic_patterns(self, text: str) -> float:
        """Detecta padrões formulaicos - versão segura"""
        if not text:
            return 0.0
        try:
            formulaic_patterns = [
                r'é importante (notar|observar|considerar|lembrar)',
                r'pode(mos)? (dizer|afirmar|concluir) que',
                r'de (acordo|forma|modo) geral',
                r'em (outras|simples|diferentes) palavras',
                r'por (outro|um) lado',
                r'neste (contexto|sentido|caso)'
            ]
            
            formula_count = 0
            for pattern in formulaic_patterns:
                try:
                    matches = re.findall(pattern, text.lower())
                    formula_count += len(matches)
                except:
                    continue
            
            sentences = len([s for s in text.split('.') if s.strip()])
            if sentences > 0:
                return min(formula_count / sentences, 1.0)
            return 0.0
        except:
            return 0.0

    def _calculate_lexical_originality(self, text: str) -> float:
        """Calcula originalidade lexical - versão segura"""
        if not text:
            return 0.0
        try:
            words = text.lower().split()
            if not words:
                return 0.0
                
            unique_words = len(set(words))
            total_words = len(words)
            lexical_diversity = unique_words / total_words
            
            common_words = {'o', 'a', 'de', 'que', 'e', 'do', 'da', 'em', 'um', 'para',
                           'é', 'com', 'não', 'uma', 'os', 'no', 'se', 'na', 'por', 'mais'}
            
            common_count = sum(1 for word in words if word in common_words)
            common_ratio = common_count / total_words
            
            originality = lexical_diversity * (1.0 - common_ratio * 0.5)
            return min(originality, 1.0)
        except:
            return 0.0

    def _detect_academic_patterns(self, text: str) -> float:
        """Detecta padrões acadêmicos - versão segura"""
        if not text:
            return 0.0
        try:
            academic_patterns = [
                r'(segundo|conforme|de acordo com).{1,30}(teoria|modelo|abordagem)',
                r'(estudos|pesquisas|análises) (mostram|indicam|sugerem|revelam)',
                r'(primeira|segunda|terceira|última) (instância|análise|consideração)',
                r'(conceito|noção|ideia) (fundamental|central|básico|essencial)',
                r'(framework|estrutura) (teórico|conceitual|analítico)'
            ]
            
            pattern_count = 0
            for pattern in academic_patterns:
                try:
                    matches = re.findall(pattern, text.lower())
                    pattern_count += len(matches)
                except:
                    continue
            
            sentences = len([s for s in text.split('.') if s.strip()])
            if sentences > 0:
                return min(pattern_count / sentences * 2, 1.0)
            return 0.0
        except:
            return 0.0

    def _analyze_stylistic_authenticity(self, text: str, rho: np.ndarray) -> float:
        """Analisa autenticidade estilística - versão segura"""
        if not text or len(rho) == 0:
            return 0.5
        try:
            text_entropy = self._calculate_text_entropy(text)
            rho_entropy = entropy(rho + 1e-10)
            
            max_text_entropy = np.log(len(text.split()) + 1)
            max_rho_entropy = np.log(len(rho))
            
            if max_text_entropy > 0 and max_rho_entropy > 0:
                text_entropy_norm = text_entropy / max_text_entropy
                rho_entropy_norm = rho_entropy / max_rho_entropy
                authenticity = 1.0 - abs(text_entropy_norm - rho_entropy_norm)
            else:
                authenticity = 0.5
            
            return max(min(authenticity, 1.0), 0.0)
        except:
            return 0.5

    def _calculate_text_entropy(self, text: str) -> float:
        """Calcula entropia do texto - versão segura"""
        if not text:
            return 0.0
        try:
            words = text.lower().split()
            if not words:
                return 0.0
                
            word_counts = Counter(words)
            total_words = len(words)
            probabilities = np.array(list(word_counts.values())) / total_words
            
            return entropy(probabilities + 1e-10)
        except:
            return 0.0

    def _calculate_jargon_substance_ratio(self, text: str) -> float:
        """Calcula ratio jargão/substância - versão segura"""
        if not text:
            return 0.0
        try:
            words = text.lower().split()
            if not words:
                return 0.0
                
            jargon_count = sum(1 for word in words 
                              if word in self.known_patterns.get("formal_indicators", set()))
            substance_count = self._calculate_substantive_content(text) * len(words)
            
            if substance_count > 0:
                ratio = jargon_count / substance_count
                return min(ratio, 1.0)
            else:
                return min(jargon_count / len(words) * 3, 1.0)
        except:
            return 0.0

    def _perform_meta_analysis(self, diagnostics: dict, model_output: dict) -> dict:
        """Meta-análise robusta"""
        try:
            scores = [v for k, v in diagnostics.items() if isinstance(v, (int, float))]
            if not scores:
                return {"error": "No valid diagnostic scores"}
                
            dominant_issue = max(diagnostics.keys(), key=lambda k: diagnostics.get(k, 0) if isinstance(diagnostics.get(k), (int, float)) else 0)
            avg_score = np.mean(scores)
            
            return {
                "dominant_issue": dominant_issue,
                "dominant_score": diagnostics.get(dominant_issue, 0),
                "average_pathology": avg_score,
                "severity_level": "high" if avg_score > 0.7 else "medium" if avg_score > 0.4 else "low",
                "stability": self._calculate_diagnostic_stability()
            }
        except Exception as e:
            return {"error": str(e)}

    def _calculate_overall_health(self, diagnostics: dict) -> float:
        """Calcula saúde geral - versão robusta corrigida"""
        
        # Pesos diferentes para cada patologia
        weights_map = {
            "obsessive_convergence": 0.3,
            "reflexive_degeneration": 0.4,  # Mais crítico
            "spurious_estetization": 0.2,
            "mimetic_resonance": 0.1
        }
        
        # Coleta apenas diagnósticos que existem e são numéricos
        existing_diagnostics = []
        existing_weights = []
        
        for key, weight in weights_map.items():
            if key in diagnostics and isinstance(diagnostics[key], (int, float)):
                existing_diagnostics.append(diagnostics[key])
                existing_weights.append(weight)
        
        if not existing_diagnostics:
            return 0.5  # Default se nenhum diagnóstico disponível
        
        # Normaliza pesos
        existing_weights = np.array(existing_weights)
        existing_weights = existing_weights / np.sum(existing_weights)
        
        # Calcula score ponderado
        weighted_score = np.average(existing_diagnostics, weights=existing_weights)
        
        # Saúde é o inverso da patologia
        health = 1.0 - weighted_score
        return max(health, 0.0)

    def _generate_recommendations(self, diagnostics: dict) -> list:
        """Gera recomendações - versão segura"""
        try:
            recommendations = []
            
            for issue, threshold_key in [
                ("obsessive_convergence", "obsessive_convergence"),
                ("reflexive_degeneration", "reflexive_degeneration"), 
                ("spurious_estetization", "spurious_estetization"),
                ("mimetic_resonance", "mimetic_resonance")
            ]:
                score = diagnostics.get(issue, 0)
                threshold = self.thresholds.get(threshold_key, 0.7)
                
                if isinstance(score, (int, float)) and score > threshold:
                    action_map = {
                        "obsessive_convergence": "Aumentar temperatura e diversificar prompts",
                        "reflexive_degeneration": "Reformular prompt com foco direto e específico",
                        "spurious_estetization": "Enfatizar conteúdo substantivo sobre forma",
                        "mimetic_resonance": "Promover originalidade e autenticidade causal"
                    }
                    
                    priority_map = {
                        "reflexive_degeneration": "critical",
                        "obsessive_convergence": "high"
                    }
                    
                    recommendations.append({
                        "issue": issue.replace("_", " ").title(),
                        "action": action_map.get(issue, "Revisar resposta"),
                        "priority": priority_map.get(issue, "medium")
                    })
            
            return recommendations
        except:
            return [{"issue": "Erro", "action": "Verificar sistema", "priority": "critical"}]

    def _analyze_temporal_trend(self, current_diagnostics: dict) -> dict:
        """Analisa tendência temporal - versão segura"""
        try:
            if len(self.diagnosis_history) < 3:
                return {"trend": "insufficient_data", "direction": "unknown"}
            
            recent_history = self.diagnosis_history[-3:]
            trends = {}
            
            for key in current_diagnostics.keys():
                if key not in ["meta_analysis", "overall_health", "recommendations", "temporal_trend"]:
                    current_value = current_diagnostics.get(key, 0)
                    if not isinstance(current_value, (int, float)):
                        continue
                        
                    past_values = []
                    for entry in recent_history:
                        if isinstance(entry, dict) and key in entry:
                            val = entry[key]
                            if isinstance(val, (int, float)):
                                past_values.append(val)
                    
                    if len(past_values) >= 2:
                        recent_avg = np.mean(past_values)
                        if current_value > recent_avg * 1.2:
                            trends[key] = "worsening"
                        elif current_value < recent_avg * 0.8:
                            trends[key] = "improving"
                        else:
                            trends[key] = "stable"
            
            if not trends:
                return {"trend": "insufficient_data"}
                
            worsening_count = sum(1 for trend in trends.values() if trend == "worsening")
            improving_count = sum(1 for trend in trends.values() if trend == "improving")
            
            if worsening_count > improving_count:
                overall_trend = "deteriorating"
            elif improving_count > worsening_count:
                overall_trend = "improving" 
            else:
                overall_trend = "stable"
            
            return {
                "trend": overall_trend,
                "individual_trends": trends,
                "change_rate": abs(worsening_count - improving_count) / len(trends)
            }
        except:
            return {"trend": "error"}

    def _calculate_diagnostic_stability(self) -> float:
        """Calcula estabilidade - versão segura"""
        try:
            if len(self.diagnosis_history) < 2:
                return 0.5
            
            diagnostic_keys = ["obsessive_convergence", "reflexive_degeneration", 
                              "spurious_estetization", "mimetic_resonance"]
            
            stabilities = []
            for key in diagnostic_keys:
                values = []
                for entry in self.diagnosis_history[-5:]:
                    if isinstance(entry, dict) and key in entry:
                        val = entry[key]
                        if isinstance(val, (int, float)):
                            values.append(val)
                
                if len(values) >= 2:
                    stability = 1.0 / (1.0 + np.std(values))
                    stabilities.append(stability)
            
            return np.mean(stabilities) if stabilities else 0.5
        except:
            return 0.5

    def _update_history(self, report: dict):
        """Atualiza histórico - versão segura"""
        try:
            history_entry = {}
            for key, value in report.items():
                if key not in ["meta_analysis", "recommendations", "temporal_trend"]:
                    if isinstance(value, (int, float, str, bool)):
                        history_entry[key] = value
            
            history_entry["timestamp"] = len(self.diagnosis_history)
            self.diagnosis_history.append(history_entry)
            
            if len(self.diagnosis_history) > self.max_history:
                self.diagnosis_history.pop(0)
        except:
            pass  # Falha silenciosa para não quebrar o sistema

    def _print_diagnostic_report(self, report: dict):
        """Imprime relatório - versão segura"""
        try:
            print("\n" + "="*60)
            print("🔍 RELATÓRIO DE DIAGNÓSTICO ONTOLÓGICO v2.1")
            print("="*60)
            
            print("\n📊 SCORES DE PATOLOGIA:")
            for key, value in report.items():
                if key not in ["meta_analysis", "overall_health", "recommendations", "temporal_trend"]:
                    if isinstance(value, (int, float)):
                        status = "🔴 CRÍTICO" if value > 0.7 else "🟡 ATENÇÃO" if value > 0.4 else "🟢 OK"
                        print(f"   {key:25}: {value:.3f} {status}")
            
            health = report.get("overall_health", 0)
            if isinstance(health, (int, float)):
                health_status = "🟢 SAUDÁVEL" if health > 0.7 else "🟡 MODERADO" if health > 0.4 else "🔴 CRÍTICO"
                print(f"\n💚 SAÚDE GERAL: {health:.3f} {health_status}")
            
            print("="*60)
        except:
            print("⚠️  Erro ao imprimir relatório diagnóstico")

    def reset_history(self):
        """Reseta histórico - versão segura"""
        try:
            self.diagnosis_history = []
            if getattr(config, 'ENABLE_DEBUG_LOGGING', False):
                print("🔍 Histórico de diagnósticos resetado")
        except:
            pass