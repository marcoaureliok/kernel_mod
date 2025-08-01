# kernel/kernel.py - Kernel Corrigido para Compatibilidade v2.1
# Versão atualizada que funciona com o novo sistema de mapeamento

import numpy as np
from engine.data_mapping import get_mapper
from kernel.equations import calculate_omega, calculate_C1
from kernel.operators import DiagnosticOperators
import config

class OntologicalKernel:
    """
    Kernel Ontológico Diferencial v2.1 - Versão Corrigida
    Implementa o ciclo completo: Monitoramento → Mapeamento → Diagnóstico → Regulação
    """
    
    def __init__(self, config_module):
        """Inicializa o kernel com configurações"""
        self.config = config_module
        self.mapper = get_mapper()
        self.operators = DiagnosticOperators()
        
        # Parâmetros do modelo ontológico
        self.omega_basal = config_module.OMEGA_BASAL
        self.lambda_dissolution = config_module.LAMBDA_DISSOLUTION
        self.mu_dissolution = config_module.MU_DISSOLUTION
        self.nu_dissolution = config_module.NU_DISSOLUTION
        self.c_max = config_module.C_MAX
        
        # Dimensões e limiares
        self.rho_dimension = config_module.RHO_SPACE_DIMENSION
        self.xi_threshold = config_module.XI_THRESHOLD
        
        if config_module.ENABLE_DEBUG_LOGGING:
            print("🧠 Kernel Ontológico v2.1 inicializado")
            print(f"   - Dimensão ρ: {self.rho_dimension}")
            print(f"   - Ω basal: {self.omega_basal}")
            print(f"   - C máx: {self.c_max}")

    def run_cycle(self, model_output):
        """
        Executa ciclo completo do kernel ontológico
        
        Args:
            model_output: Pode ser dict (formato novo) ou string (compatibilidade)
            
        Returns:
            dict: Resultado completo do ciclo ontológico
        """
        
        # === COMPATIBILIDADE: Converte string para formato esperado ===
        if isinstance(model_output, str):
            # Formato antigo - converte para novo formato
            normalized_output = {
                "text": model_output,
                "logits": np.array([]),
                "tokens": [],
                "generation_stats": {"tokens_generated": len(model_output.split())},
                "internal_states": {"perplexity": 1.0}
            }
            if self.config.ENABLE_DEBUG_LOGGING:
                print("🔄 Convertendo entrada string para formato dict (compatibilidade)")
        elif isinstance(model_output, dict):
            # Formato novo - usa diretamente
            normalized_output = model_output
        else:
            # Fallback para casos inesperados
            normalized_output = {
                "text": str(model_output),
                "logits": np.array([]),
                "tokens": [],
                "generation_stats": {},
                "internal_states": {}
            }
            print(f"⚠️  Formato de entrada inesperado: {type(model_output)}")
        
        try:
            # === FASE 1: MAPEAMENTO ONTOLÓGICO ===
            if self.config.LOG_KERNEL_STATE:
                print("🗺️  Executando mapeamento ontológico ρ → E...")
            
            # Mapeia para campo propensional ρ(x,t)
            rho = self.mapper.map_to_rho(normalized_output, self.rho_dimension)
            
            # Mapeia para campo reflexivo E(x,t)
            E = self.mapper.map_to_E(normalized_output, self.rho_dimension, self.c_max)
            
            if self.config.ENABLE_DEBUG_LOGGING:
                print(f"   ρ: shape={rho.shape}, sum={np.sum(rho):.4f}, entropy={self._calculate_entropy(rho):.4f}")
                print(f"   E: shape={E.shape}, norm={np.linalg.norm(E):.4f}, max={np.max(E):.4f}")
            
            # === FASE 2: CÁLCULOS DAS EQUAÇÕES FUNDAMENTAIS ===
            if self.config.LOG_KERNEL_STATE:
                print("🧮 Calculando equações ontológicas...")
            
            # Calcula métricas derivadas
            A_rho = np.sum(rho * np.arange(len(rho)))  # Momento de ρ
            E_psi = np.linalg.norm(E)  # Magnitude reflexiva
            S_psi = np.std(E)  # Dispersão reflexiva
            
            # Calcula Ω(t) - gradiente de dissolução (EMF-1)
            omega = calculate_omega(
                A_rho=A_rho,
                E_psi=E_psi,
                omega_basal=self.omega_basal,
                lambda_diss=self.lambda_dissolution,
                mu_diss=self.mu_dissolution,
                nu_diss=self.nu_dissolution
            )
            
            # Calcula C₁(t) - intensificação reflexiva (EMF-5)
            c1 = calculate_C1(
                E_psi=E_psi,
                S_psi=S_psi,
                time_step=1.0,  # Normalizado para esta iteração
                k_sigmoid=self.config.K_SIGMOID,
                e_threshold=self.config.E_THRESHOLD,
                c_max=self.c_max,
                beta_rhythm=self.config.BETA_RHYTHM,
                nu_e_freq=self.config.NU_E_FREQUENCY,
                phi_phase=self.config.PHI_PHASE
            )
            
            if self.config.ENABLE_DEBUG_LOGGING:
                print(f"   Ω (dissolução): {omega:.4f}")
                print(f"   C₁ (intensidade): {c1:.4f}")
                print(f"   A_ρ (momento): {A_rho:.4f}")
                print(f"   𝔈_ψ (reflexão): {E_psi:.4f}")
            
            # === FASE 3: DIAGNÓSTICO ONTOLÓGICO ===
            if self.config.LOG_KERNEL_STATE:
                print("🔍 Executando diagnósticos ontológicos...")
            
            # Executa todos os operadores de diagnóstico
            diagnostics = self.operators.run_all_diagnostics(rho, E, normalized_output)
            
            if self.config.ENABLE_DEBUG_LOGGING:
                dominant_issue = diagnostics.get("meta_analysis", {}).get("dominant_issue", "N/A")
                overall_health = diagnostics.get("overall_health", 0)
                print(f"   Problema dominante: {dominant_issue}")
                print(f"   Saúde geral: {overall_health:.3f}")
            
            # === FASE 4: REGULAÇÃO ONTOLÓGICA ===
            regulation_output = {
                "omega": omega,
                "c1": c1,
                "A_rho": A_rho,
                "E_psi": E_psi,
                "S_psi": S_psi,
                "requires_intervention": self._assess_intervention_need(diagnostics, omega, c1)
            }
            
            # === RESULTADO FINAL ===
            return {
                "fields": {
                    "rho": rho,
                    "E": E
                },
                "metrics": {
                    "omega": omega,
                    "c1": c1,
                    "A_rho": A_rho,
                    "E_psi": E_psi,
                    "S_psi": S_psi
                },
                "diagnostics": diagnostics,
                "regulation": regulation_output,
                "kernel_state": {
                    "iteration": getattr(self, '_iteration_count', 0),
                    "stability": self._calculate_stability(rho, E),
                    "coherence": self._calculate_coherence(rho, E)
                }
            }
            
        except Exception as e:
            print(f"❌ Erro no ciclo do kernel: {e}")
            if self.config.ENABLE_DEBUG_LOGGING:
                import traceback
                traceback.print_exc()
            
            # Retorna estado de erro mas funcional
            return {
                "fields": {
                    "rho": np.ones(self.rho_dimension) / self.rho_dimension,
                    "E": np.zeros(self.rho_dimension)
                },
                "metrics": {
                    "omega": self.omega_basal,
                    "c1": self.c_max * 0.5,
                    "A_rho": 0.5,
                    "E_psi": 0.0,
                    "S_psi": 0.0
                },
                "diagnostics": {"error": str(e), "overall_health": 0.1},
                "regulation": {
                    "omega": self.omega_basal,
                    "c1": self.c_max * 0.5,
                    "requires_intervention": True
                },
                "kernel_state": {
                    "iteration": 0,
                    "stability": 0.0,
                    "coherence": 0.0,
                    "error": str(e)
                }
            }

    def _assess_intervention_need(self, diagnostics, omega, c1):
        """Avalia se intervenção regulatória é necessária"""
        
        # Critérios de intervenção
        criteria = []
        
        # Alta dissolução
        if omega > self.omega_basal * 3:
            criteria.append("high_dissolution")
        
        # Baixa intensidade reflexiva
        if c1 < self.c_max * 0.3:
            criteria.append("low_reflexive_intensity")
        
        # Saúde geral baixa
        overall_health = diagnostics.get("overall_health", 1.0)
        if overall_health < 0.5:
            criteria.append("poor_health")
        
        # Patologia dominante severa
        meta = diagnostics.get("meta_analysis", {})
        if meta.get("severity_level") in ["high", "critical"]:
            criteria.append("severe_pathology")
        
        return {
            "needed": len(criteria) > 0,
            "criteria": criteria,
            "urgency": "critical" if len(criteria) >= 3 else "moderate" if len(criteria) >= 2 else "low"
        }

    def _calculate_entropy(self, field):
        """Calcula entropia de um campo"""
        if len(field) == 0 or np.sum(field) == 0:
            return 0.0
        
        # Normaliza para probabilidade
        p = field / np.sum(field)
        p = p[p > 0]  # Remove zeros
        
        return -np.sum(p * np.log(p))

    def _calculate_stability(self, rho, E):
        """Calcula estabilidade do sistema baseada nos campos"""
        if len(rho) == 0 or len(E) == 0:
            return 0.0
        
        # Estabilidade baseada na suavidade dos campos
        rho_stability = 1.0 / (1.0 + np.std(np.diff(rho)))
        E_stability = 1.0 / (1.0 + np.std(np.diff(E)))
        
        return (rho_stability + E_stability) / 2

    def _calculate_coherence(self, rho, E):
        """Calcula coerência entre os campos ρ e E"""
        if len(rho) == 0 or len(E) == 0 or len(rho) != len(E):
            return 0.0
        
        try:
            # Normaliza ambos os campos
            rho_norm = rho / np.sum(rho) if np.sum(rho) > 0 else rho
            E_norm = E / np.linalg.norm(E) if np.linalg.norm(E) > 0 else E
            
            # Calcula correlação como medida de coerência
            correlation = np.corrcoef(rho_norm, E_norm)[0, 1]
            
            # Trata NaN (quando um dos campos é constante)
            if np.isnan(correlation):
                return 0.5
            
            # Converte correlação [-1,1] para coerência [0,1]
            coherence = (correlation + 1) / 2
            
            return coherence
        except:
            return 0.5

    def get_kernel_info(self):
        """Retorna informações sobre o estado do kernel"""
        
        return {
            "version": "2.1",
            "rho_dimension": self.rho_dimension,
            "omega_basal": self.omega_basal,
            "c_max": self.c_max,
            "mapper_stats": self.mapper.get_mapping_stats(),
            "diagnostic_history_length": len(self.operators.diagnosis_history),
            "configuration": {
                "lambda_dissolution": self.lambda_dissolution,
                "mu_dissolution": self.mu_dissolution,
                "nu_dissolution": self.nu_dissolution,
                "xi_threshold": self.xi_threshold
            }
        }

    def reset_kernel_state(self):
        """Reseta o estado interno do kernel"""
        
        # Reseta mapeador
        self.mapper.reset_history()
        
        # Reseta operadores de diagnóstico
        self.operators.reset_history()
        
        # Reseta contador de iterações se existir
        if hasattr(self, '_iteration_count'):
            self._iteration_count = 0
        
        if self.config.ENABLE_DEBUG_LOGGING:
            print("🔄 Estado do kernel resetado")

    def calibrate_thresholds(self, calibration_data=None):
        """Calibra limiares do sistema baseado em dados históricos"""
        
        if not calibration_data and len(self.operators.diagnosis_history) < 10:
            print("⚠️  Dados insuficientes para calibração automática")
            return False
        
        try:
            # Usa histórico existente se não fornecido dados específicos
            if not calibration_data:
                calibration_data = self.operators.diagnosis_history
            
            # Extrai scores históricos
            pathology_scores = {
                "obsessive_convergence": [],
                "reflexive_degeneration": [],
                "spurious_estetization": [],
                "mimetic_resonance": []
            }
            
            for entry in calibration_data:
                for pathology in pathology_scores.keys():
                    if pathology in entry:
                        pathology_scores[pathology].append(entry[pathology])
            
            # Recalcula limiares baseado em percentis
            new_thresholds = {}
            for pathology, scores in pathology_scores.items():
                if len(scores) >= 5:  # Mínimo de dados
                    # Usa 75º percentil como limiar (captura outliers significativos)
                    new_threshold = np.percentile(scores, 75)
                    new_thresholds[pathology] = min(max(new_threshold, 0.3), 0.9)  # Limita entre 0.3-0.9
            
            # Atualiza limiares se encontrou dados suficientes
            if new_thresholds:
                self.operators.thresholds.update(new_thresholds)
                
                print("🎯 Limiares calibrados automaticamente:")
                for pathology, threshold in new_thresholds.items():
                    print(f"   {pathology}: {threshold:.3f}")
                
                return True
            else:
                print("⚠️  Calibração falhou - dados insuficientes")
                return False
                
        except Exception as e:
            print(f"❌ Erro na calibração: {e}")
            return False
