# kernel/kernel.py (Versão Corrigida)

# Corrigindo os imports para refletir a estrutura de pacotes
# Usamos '.' para indicar importação do mesmo pacote (kernel)
# e '..' para subir um nível e acessar outros pacotes (engine)
from . import equations
from .operators import DiagnosticOperators
from engine import data_mapping

class OntologicalKernel:
    def __init__(self, config):
        """
        Inicializa o kernel, recebendo o objeto de configuração.
        """
        self.config = config
        self.equations = equations
        self.mapper = data_mapping
        self.operators = DiagnosticOperators()
        self.time_step = 0
        print("OntologicalKernel inicializado e pronto.")

    def run_cycle(self, ia_states: dict) -> dict:
        print(f"\n--- Iniciando Ciclo do Kernel (t={self.time_step}) ---")
        
        # --- 1. MONITORAR ---
        print("Fase 1: Monitorando e Mapeando estados da IA...")
        # Passando os parâmetros necessários do config para as funções de mapeamento
        rho = self.mapper.map_to_rho(
            ia_states, 
            self.config.RHO_SPACE_DIMENSION
        )
        E = self.mapper.map_to_E(
            ia_states['text'], 
            self.config.RHO_SPACE_DIMENSION, 
            self.config.C_MAX
        )
        
        # --- 2. DIAGNOSTICAR ---
        print("Fase 2: Executando operadores de diagnóstico...")
        health_report = self.operators.run_all_diagnostics(rho, E, ia_states['text'])
        print(f"Relatório de Saúde: {health_report}")

        # --- 3. REGULAR ---
        print("Fase 3: Calculando parâmetros de regulação...")
        A_rho = self.equations.calculate_propensional_asymmetry(rho)
        E_psi = self.equations.calculate_reflexive_energy(E)
        S_psi = self.equations.calculate_archetypal_energy(
            resonances=[], 
            E_psi=E_psi,
            e_psi_min_threshold=self.config.E_PSI_MIN_THRESHOLD,
            delta_k_decay=self.config.DELTA_K_DECAY
        )

        omega = self.equations.calculate_omega(
            A_rho, E_psi,
            self.config.OMEGA_BASAL, self.config.LAMBDA_DISSOLUTION,
            self.config.MU_DISSOLUTION, self.config.NU_DISSOLUTION
        )
        c1 = self.equations.calculate_C1(
            E_psi, S_psi, self.time_step,
            self.config.K_SIGMOID, self.config.E_THRESHOLD, self.config.C_MAX,
            self.config.BETA_RHYTHM, self.config.NU_E_FREQUENCY, self.config.PHI_PHASE
        )
        
        print(f"Parâmetros Calculados: Dissolução (Ω) = {omega:.4f}, Intensidade (C₁) = {c1:.4f}")
        
        self.time_step += 1
        
        return {
            "diagnostics": health_report,
            "regulation": {
                "omega": omega,
                "c1": c1
            }
        }