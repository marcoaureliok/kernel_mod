# main.py - Sistema Principal Corrigido v2.1
# Kernel Ontológico Diferencial totalmente funcional com todos os problemas corrigidos

import sys
import traceback
from pathlib import Path
import numpy as np

# Importações do sistema
import config
from kernel.kernel import OntologicalKernel
from engine.local_model_handler import LocalModelHandler

class OntologicalSystem:
    """Sistema principal do Kernel Ontológico v2.1"""
    
    def __init__(self):
        self.model_handler = None
        self.kernel = None
        self.session_stats = {
            "interactions": 0,
            "refinements_triggered": 0,
            "total_generation_time": 0,
            "avg_health_score": 0,
            "dominant_pathologies": {}
        }
        self.regulation_params = {
            "temperature": config.LLM_TEMPERATURE,
            "top_p": config.LLM_TOP_P,
            "top_k": config.LLM_TOP_K,
            "repetition_penalty": config.LLM_REPETITION_PENALTY
        }
        
        self._initialize_system()

    def _initialize_system(self):
        """Inicializa todos os componentes do sistema"""
        
        print("🚀 INICIANDO KERNEL ONTOLÓGICO DIFERENCIAL v2.1")
        print("="*70)
        
        # Validação da configuração
        if config.CONFIG_ISSUES:
            print("⚠️  PROBLEMAS DE CONFIGURAÇÃO DETECTADOS:")
            for issue in config.CONFIG_ISSUES:
                print(f"   ❌ {issue}")
            print("\n💡 Continue mesmo assim? O sistema tentará funcionar com fallbacks.")
            
            response = input("Continuar? (s/n): ").lower()
            if response != 's':
                print("❌ Sistema cancelado pelo usuário.")
                sys.exit(1)
        
        # Inicializa handler do modelo
        try:
            print("\n🤖 Inicializando handler do modelo...")
            self.model_handler = LocalModelHandler()
            
            # Verifica se o modelo foi carregado
            model_info = self.model_handler.get_model_info()
            if model_info["status"] == "loaded":
                print("✅ Modelo carregado com sucesso!")
                print(f"   📂 Arquivo: {Path(model_info['model_path']).name}")
                print(f"   🔧 GPU Layers: {model_info['gpu_layers']}")
                print(f"   📏 Contexto: {model_info['context_size']} tokens")
            else:
                print("⚠️  Modelo não carregado - usando modo API/fallback")
                
        except Exception as e:
            print(f"❌ Erro ao inicializar modelo: {e}")
            print("🔄 Sistema continuará em modo degradado")
            traceback.print_exc()
        
        # Inicializa kernel ontológico
        try:
            print("\n🧠 Inicializando kernel ontológico...")
            self.kernel = OntologicalKernel(config)
            print("✅ Kernel ontológico inicializado!")
            
        except Exception as e:
            print(f"❌ Erro crítico ao inicializar kernel: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        print("\n" + "="*70)
        print("🎯 SISTEMA PRONTO PARA OPERAÇÃO")
        print("   Digite 'sair' para encerrar")
        print("   Digite 'status' para ver estatísticas")
        print("   Digite 'reset' para limpar histórico")
        print("="*70)

    def apply_regulation(self, regulation_output: dict) -> dict:
        """
        Aplica regulação ontológica aos parâmetros de geração - VERSÃO CORRIGIDA
        """
        
        regulation = regulation_output.get('regulation', {})
        omega = regulation.get('omega', config.OMEGA_BASAL)
        c1 = regulation.get('c1', config.C_MAX * 0.5)
        
        # Fórmula de regulação calibrada
        base_temp = config.LLM_TEMPERATURE
        temp_adjustment = (omega * 1.5) - (c1 * 0.4)
        new_temperature = base_temp + temp_adjustment
        
        # *** CORREÇÃO CRÍTICA: Limites seguros mais altos ***
        new_temperature = max(0.3, min(new_temperature, 1.2))  # MÍNIMO 0.3 (era 0.1)
        
        # Ajustes para outros parâmetros
        if omega > 0.3:
            new_top_p = min(config.LLM_TOP_P + 0.1, 0.95)
            new_top_k = min(config.LLM_TOP_K + 10, 50)
        else:
            new_top_p = max(config.LLM_TOP_P - 0.1, 0.8)  # MÍNIMO 0.8 (era 0.7)
            new_top_k = max(config.LLM_TOP_K - 5, 25)    # MÍNIMO 25 (era 20)
        
        # Alta intensidade (c1) -> menos repetição
        if c1 > 0.7:
            new_repetition_penalty = min(config.LLM_REPETITION_PENALTY + 0.1, 1.25)  # MÁXIMO 1.25 (era 1.3)
        else:
            new_repetition_penalty = config.LLM_REPETITION_PENALTY
        
        regulated_params = {
            "temperature": new_temperature,
            "top_p": new_top_p,
            "top_k": int(new_top_k),
            "repetition_penalty": new_repetition_penalty
        }
        
        if config.LOG_REGULATION_ACTIONS:
            print(f"\n🎛️  AÇÃO REGULATÓRIA DO KERNEL:")
            print(f"   🌀 Dissolução (Ω): {omega:.3f}")
            print(f"   ⚡ Intensidade (C₁): {c1:.3f}")
            print(f"   🌡️  Temperatura: {base_temp:.2f} → {new_temperature:.2f}")
            print(f"   🎯 Top-p: {config.LLM_TOP_P:.2f} → {new_top_p:.2f}")
            print(f"   🔢 Top-k: {config.LLM_TOP_K} → {int(new_top_k)}")
            print(f"   🔄 Rep. Penalty: {config.LLM_REPETITION_PENALTY:.2f} → {new_repetition_penalty:.2f}")
        
        return regulated_params



    def create_corrective_prompt(self, original_prompt: str, diagnostics: dict, 
                               attempt: int) -> str:
        """
        Cria prompt corretivo baseado nos diagnósticos
        
        Args:
            original_prompt: Prompt original do usuário
            diagnostics: Resultado dos diagnósticos
            attempt: Número da tentativa (para escalação)
            
        Returns:
            str: Prompt corretivo
        """
        
        # Identifica o problema dominante
        dominant_issue = diagnostics.get("meta_analysis", {}).get("dominant_issue", "")
        dominant_score = diagnostics.get(dominant_issue, 0)
        
        corrections = []
        
        # Correções específicas por patologia
        if dominant_issue == "reflexive_degeneration" and dominant_score > 0.6:
            if attempt == 1:
                corrections.append("Responda de forma direta e objetiva, sem evasivas ou metacomentários.")
            else:
                corrections.append("IMPORTANTE: Sua resposta anterior foi evasiva. Responda DIRETAMENTE à pergunta específica.")
        
        if dominant_issue == "obsessive_convergence" and dominant_score > 0.7:
            corrections.append("Varie sua abordagem e evite repetições. Explore diferentes aspectos da questão.")
        
        if dominant_issue == "spurious_estetization" and dominant_score > 0.5:
            corrections.append("Priorize conteúdo substantivo sobre forma. Seja claro e prático.")
        
        if dominant_issue == "mimetic_resonance" and dominant_score > 0.8:
            corrections.append("Seja original e autêntico. Evite fórmulas pré-fabricadas.")
        
        # Monta o prompt corretivo
        if corrections:
            correction_text = " ".join(corrections)
            corrective_prompt = f"{correction_text}\n\nPergunta original: {original_prompt}"
        else:
            # Fallback genérico
            corrective_prompt = f"Reformule sua abordagem anterior. Seja mais direto e específico.\n\nPergunta: {original_prompt}"
        
        if config.ENABLE_DEBUG_LOGGING:
            print(f"🔧 PROMPT CORRETIVO (tentativa {attempt}):")
            print(f"   Problema dominante: {dominant_issue} ({dominant_score:.3f})")
            print(f"   Correção aplicada: {correction_text if corrections else 'Genérica'}")
        
        return corrective_prompt

    def run_interaction_cycle(self, user_prompt: str) -> dict:
        """
        Executa um ciclo completo de interação com refinamento
        
        Args:
            user_prompt: Prompt do usuário
            
        Returns:
            dict: Resultado da interação com metadados
        """
        
        self.session_stats["interactions"] += 1
        interaction_start_time = 0  # Placeholder para timing
        
        print(f"\n{'='*50}")
        print(f"💬 INTERAÇÃO #{self.session_stats['interactions']}")
        print(f"{'='*50}")
        
        # --- FASE 1: ANÁLISE PRÉ-GERAÇÃO ---
        if config.LOG_KERNEL_STATE:
            print("\n🔍 FASE 1: Análise Pré-Geração (Análise do Prompt)")
        
        try:
            # Cria "estado" do prompt para análise
            prompt_analysis = {
                "text": user_prompt,
                "logits": np.array([]),  # Será preenchido pelo mapeador
                "internal_states": {"perplexity": 1.0},
                "generation_stats": {"tokens_generated": len(user_prompt.split())}
            }
            
            # Executa ciclo do kernel no prompt
            pre_control_output = self.kernel.run_cycle(prompt_analysis)
            
            # Aplica regulação inicial
            initial_regulation = self.apply_regulation(pre_control_output)
            self.regulation_params.update(initial_regulation)
            
        except Exception as e:
            print(f"⚠️  Erro na análise pré-geração: {e}")
            if config.ENABLE_DEBUG_LOGGING:
                traceback.print_exc()
        
        # --- FASE 2: CICLO DE GERAÇÃO E REFINAMENTO ---
        attempts = 0
        final_response = ""
        final_diagnostics = {}
        refinement_history = []
        
        current_prompt = user_prompt
        
        while attempts <= config.MAX_REFINEMENT_ATTEMPTS:
            attempts += 1
            
            if config.LOG_KERNEL_STATE:
                print(f"\n🤖 FASE 2.{attempts}: Geração" + (f" (Refinamento)" if attempts > 1 else ""))
            
            try:
                # Gera resposta
                model_output = self.model_handler.generate_response(
                    current_prompt, **self.regulation_params
                )
                
                if not model_output or not model_output.get("text"):
                    print("❌ Falha na geração. Tentando novamente...")
                    continue
                
                generated_text = model_output["text"]
                
                if config.ENABLE_DEBUG_LOGGING:
                    print(f"📝 Texto gerado ({len(generated_text)} chars): {generated_text[:100]}...")
                
            except Exception as e:
                print(f"❌ Erro na geração (tentativa {attempts}): {e}")
                if attempts > config.MAX_REFINEMENT_ATTEMPTS:
                    final_response = f"Erro na geração após {attempts} tentativas: {str(e)}"
                    break
                continue
            
            # --- FASE 3: ANÁLISE PÓS-GERAÇÃO ---
            if config.LOG_KERNEL_STATE:
                print(f"\n🔍 FASE 3.{attempts}: Análise Pós-Geração (Diagnóstico)")
            
            try:
                # Executa diagnóstico completo
                post_control_output = self.kernel.run_cycle(model_output)
                diagnostics = post_control_output["diagnostics"]
                
                # Armazena histórico de refinamento
                refinement_entry = {
                    "attempt": attempts,
                    "text": generated_text,
                    "diagnostics": diagnostics.copy(),
                    "regulation": post_control_output["regulation"].copy()
                }
                refinement_history.append(refinement_entry)
                
                # Verifica se refinamento é necessário
                if not config.ENABLE_REFINEMENT_CYCLE:
                    # Refinamento desabilitado - aceita qualquer resposta
                    final_response = generated_text
                    final_diagnostics = diagnostics
                    if config.LOG_KERNEL_STATE:
                        print("✅ Refinamento desabilitado - resposta aceita")
                    break
                
                # Verifica critérios de aceitação
                dominant_issue = diagnostics.get("meta_analysis", {}).get("dominant_issue", "")
                dominant_score = diagnostics.get(dominant_issue, 0)
                overall_health = diagnostics.get("overall_health", 1.0)
                
                # Critérios de aceitação (mais flexíveis)
                acceptance_criteria = [
                    dominant_score < config.REFINEMENT_DEGENERATION_THRESHOLD,
                    overall_health > 0.3,  # Critério de saúde mínima
                    len(generated_text.strip()) > 10  # Resposta não vazia
                ]
                
                if all(acceptance_criteria) or attempts > config.MAX_REFINEMENT_ATTEMPTS:
                    final_response = generated_text
                    final_diagnostics = diagnostics
                    if config.LOG_KERNEL_STATE:
                        status = "✅ Resposta aprovada" if all(acceptance_criteria) else "⏰ Limite de tentativas atingido"
                        print(f"{status} - Saúde: {overall_health:.3f}")
                    break
                else:
                    # Prepara para refinamento
                    self.session_stats["refinements_triggered"] += 1
                    if config.LOG_KERNEL_STATE:
                        print(f"🔄 Refinamento necessário - Problema: {dominant_issue} ({dominant_score:.3f})")
                    
                    # Cria prompt corretivo
                    current_prompt = self.create_corrective_prompt(user_prompt, diagnostics, attempts)
                    
                    # Atualiza regulação para próxima tentativa
                    new_regulation = self.apply_regulation(post_control_output)
                    self.regulation_params.update(new_regulation)
                
            except Exception as e:
                print(f"⚠️  Erro no diagnóstico (tentativa {attempts}): {e}")
                if config.ENABLE_DEBUG_LOGGING:
                    traceback.print_exc()
                
                # Em caso de erro, aceita a resposta se ela existe
                if generated_text:
                    final_response = generated_text
                    final_diagnostics = {"error": str(e)}
                    break
        
        # --- FASE 4: ATUALIZAÇÃO FINAL E ESTATÍSTICAS ---
        if final_diagnostics:
            # Atualiza estatísticas da sessão
            health_score = final_diagnostics.get("overall_health", 0)
            self.session_stats["avg_health_score"] = (
                (self.session_stats["avg_health_score"] * (self.session_stats["interactions"] - 1) + health_score) /
                self.session_stats["interactions"]
            )
            
            # Rastreia patologias dominantes
            dominant_issue = final_diagnostics.get("meta_analysis", {}).get("dominant_issue", "unknown")
            if dominant_issue in self.session_stats["dominant_pathologies"]:
                self.session_stats["dominant_pathologies"][dominant_issue] += 1
            else:
                self.session_stats["dominant_pathologies"][dominant_issue] = 1
        
        return {
            "response": final_response,
            "attempts": attempts,
            "diagnostics": final_diagnostics,
            "refinement_history": refinement_history,
            "regulation_applied": self.regulation_params.copy(),
            "session_stats": self.session_stats.copy()
        }

    def handle_special_commands(self, user_input: str) -> bool:
        """
        Processa comandos especiais do sistema
        
        Args:
            user_input: Input do usuário
            
        Returns:
            bool: True se foi um comando especial, False caso contrário
        """
        
        command = user_input.lower().strip()
        
        if command == "status":
            self.print_system_status()
            return True
        
        elif command == "reset":
            self.reset_system()
            return True
        
        elif command == "debug":
            self.toggle_debug_mode()
            return True
        
        elif command == "model":
            self.print_model_info()
            return True
        
        elif command == "config":
            self.print_config_info()
            return True
        
        elif command == "help":
            self.print_help()
            return True
        
        return False

    def print_system_status(self):
        """Imprime status detalhado do sistema"""
        
        print("\n" + "="*60)
        print("📊 STATUS DO SISTEMA ONTOLÓGICO")
        print("="*60)
        
        # Estatísticas da sessão
        print(f"🔢 ESTATÍSTICAS DA SESSÃO:")
        print(f"   Interações: {self.session_stats['interactions']}")
        print(f"   Refinamentos: {self.session_stats['refinements_triggered']}")
        print(f"   Taxa de Refinamento: {(self.session_stats['refinements_triggered']/max(1,self.session_stats['interactions']))*100:.1f}%")
        print(f"   Saúde Média: {self.session_stats['avg_health_score']:.3f}")
        
        # Patologias dominantes
        if self.session_stats["dominant_pathologies"]:
            print(f"\n🦠 PATOLOGIAS MAIS FREQUENTES:")
            sorted_pathologies = sorted(self.session_stats["dominant_pathologies"].items(), 
                                      key=lambda x: x[1], reverse=True)
            for pathology, count in sorted_pathologies[:3]:
                print(f"   {pathology}: {count}x")
        
        # Status do modelo
        if self.model_handler:
            model_info = self.model_handler.get_model_info()
            print(f"\n🤖 MODELO:")
            print(f"   Status: {model_info['status']}")
            if model_info['status'] == 'loaded':
                gen_stats = model_info.get('generation_stats', {})
                print(f"   Gerações: {gen_stats.get('successful_generations', 0)}")
                print(f"   Taxa de Sucesso: {(gen_stats.get('successful_generations', 0)/max(1,gen_stats.get('total_generations', 1)))*100:.1f}%")
        
        # Parâmetros atuais
        print(f"\n⚙️  PARÂMETROS ATUAIS:")
        for param, value in self.regulation_params.items():
            print(f"   {param}: {value}")
        
        print("="*60)

    def reset_system(self):
        """Reseta o sistema para estado inicial"""
        
        print("\n🔄 RESETANDO SISTEMA...")
        
        # Reseta estatísticas
        self.session_stats = {
            "interactions": 0,
            "refinements_triggered": 0,
            "total_generation_time": 0,
            "avg_health_score": 0,
            "dominant_pathologies": {}
        }
        
        # Reseta parâmetros de regulação
        self.regulation_params = {
            "temperature": config.LLM_TEMPERATURE,
            "top_p": config.LLM_TOP_P,
            "top_k": config.LLM_TOP_K,
            "repetition_penalty": config.LLM_REPETITION_PENALTY
        }
        
        # Reseta históricos dos componentes
        if self.model_handler:
            self.model_handler.reset_stats()
        
        if self.kernel:
            # Reseta históricos do kernel se possível
            try:
                from engine.data_mapping import get_mapper
                get_mapper().reset_history()
                
                # Reseta histórico dos operadores
                self.kernel.operators.reset_history()
                
            except Exception as e:
                if config.ENABLE_DEBUG_LOGGING:
                    print(f"Aviso: Não foi possível resetar todos os históricos: {e}")
        
        print("✅ Sistema resetado com sucesso!")

    def toggle_debug_mode(self):
        """Alterna modo debug"""
        
        config.ENABLE_DEBUG_LOGGING = not config.ENABLE_DEBUG_LOGGING
        config.VERBOSE_DIAGNOSTICS = config.ENABLE_DEBUG_LOGGING
        config.LOG_KERNEL_STATE = config.ENABLE_DEBUG_LOGGING
        config.LOG_REGULATION_ACTIONS = config.ENABLE_DEBUG_LOGGING
        
        status = "ATIVADO" if config.ENABLE_DEBUG_LOGGING else "DESATIVADO"
        print(f"\n🐛 Modo Debug {status}")

    def print_model_info(self):
        """Imprime informações detalhadas do modelo"""
        
        if not self.model_handler:
            print("❌ Model handler não inicializado")
            return
        
        info = self.model_handler.get_model_info()
        
        print("\n" + "="*50)
        print("🤖 INFORMAÇÕES DO MODELO")
        print("="*50)
        
        for key, value in info.items():
            print(f"{key}: {value}")
        
        print("="*50)

    def print_config_info(self):
        """Imprime informações de configuração"""
        
        print("\n" + "="*50)
        print("⚙️  CONFIGURAÇÃO ATUAL")
        print("="*50)
        
        important_configs = [
            ("LOCAL_MODEL_PATH", config.LOCAL_MODEL_PATH),
            ("RHO_SPACE_DIMENSION", config.RHO_SPACE_DIMENSION),
            ("ENABLE_REFINEMENT_CYCLE", config.ENABLE_REFINEMENT_CYCLE),
            ("MAX_REFINEMENT_ATTEMPTS", config.MAX_REFINEMENT_ATTEMPTS),
            ("DEVICE", config.DEVICE),
            ("N_GPU_LAYERS", config.N_GPU_LAYERS),
        ]
        
        for name, value in important_configs:
            print(f"{name}: {value}")
        
        print("="*50)

    def print_help(self):
        """Imprime ajuda do sistema"""
        
        print("\n" + "="*50)
        print("❓ COMANDOS DISPONÍVEIS")
        print("="*50)
        print("status  - Mostra estatísticas do sistema")
        print("reset   - Reseta sistema para estado inicial")  
        print("debug   - Alterna modo debug on/off")
        print("model   - Mostra informações do modelo")
        print("config  - Mostra configuração atual")
        print("help    - Mostra esta ajuda")
        print("sair    - Encerra o sistema")
        print("="*50)

    def run(self):
        """Loop principal do sistema"""
        
        if not self.model_handler or not self.kernel:
            print("❌ Sistema não inicializado corretamente. Encerrando.")
            return
        
        try:
            while True:
                print(f"\n{'='*30}")
                user_input = input("👤 Você: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == 'sair':
                    break
                
                # Verifica comandos especiais
                if self.handle_special_commands(user_input):
                    continue
                
                # Executa ciclo de interação normal
                try:
                    result = self.run_interaction_cycle(user_input)
                    
                    # Exibe resposta
                    print(f"\n🤖 IA: {result['response']}")
                    
                    # Exibe metadados se debug ativo
                    if config.ENABLE_DEBUG_LOGGING:
                        print(f"\n📈 Tentativas: {result['attempts']}")
                        health = result['diagnostics'].get('overall_health', 0)
                        print(f"💚 Saúde: {health:.3f}")
                    
                except KeyboardInterrupt:
                    print("\n⏸️  Interação interrompida pelo usuário")
                    continue
                except Exception as e:
                    print(f"\n❌ Erro na interação: {e}")
                    if config.ENABLE_DEBUG_LOGGING:
                        traceback.print_exc()
                    continue
        
        except KeyboardInterrupt:
            print("\n\n⏸️  Sistema interrompido pelo usuário")
        
        finally:
            self.print_session_summary()

    def print_session_summary(self):
        """Imprime resumo da sessão"""
        
        print("\n" + "="*60)
        print("📋 RESUMO DA SESSÃO")
        print("="*60)
        
        print(f"Total de interações: {self.session_stats['interactions']}")
        print(f"Refinamentos disparados: {self.session_stats['refinements_triggered']}")
        
        if self.session_stats['interactions'] > 0:
            refinement_rate = (self.session_stats['refinements_triggered'] / 
                             self.session_stats['interactions']) * 100
            print(f"Taxa de refinamento: {refinement_rate:.1f}%")
            print(f"Saúde média: {self.session_stats['avg_health_score']:.3f}")
        
        if self.session_stats['dominant_pathologies']:
            print("\nPatologias mais frequentes:")
            sorted_pathologies = sorted(self.session_stats['dominant_pathologies'].items(),
                                      key=lambda x: x[1], reverse=True)
            for pathology, count in sorted_pathologies:
                print(f"  - {pathology}: {count}x")
        
        print("\n🎯 SISTEMA ONTOLÓGICO ENCERRADO")
        print("="*60)


# *** ADICIONE ESTA FUNÇÃO DE GERAÇÃO COM TIMEOUT ***
def generate_with_timeout(self, current_prompt, timeout_seconds=30):
    """Gera resposta com timeout para evitar travamentos"""
    
    import signal
    import threading
    
    result = {"success": False, "output": None, "error": None}
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Geração excedeu tempo limite")
    
    def generate_thread():
        try:
            output = self.model_handler.generate_response(
                current_prompt, **self.regulation_params
            )
            result["output"] = output
            result["success"] = True
        except Exception as e:
            result["error"] = str(e)
    
    try:
        # Configura timeout
        if hasattr(signal, 'SIGALRM'):  # Unix/Linux/Mac
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
        
        # Executa geração
        thread = threading.Thread(target=generate_thread)
        thread.daemon = True
        thread.start()
        thread.join(timeout_seconds)
        
        if thread.is_alive():
            print(f"⏰ Timeout após {timeout_seconds}s - forçando parada")
            return {
                "text": "⏰ Geração interrompida por timeout. Ajustando parâmetros...",
                "logits": np.array([]),
                "tokens": [],
                "generation_stats": {"timeout": True},
                "internal_states": {"timeout": True}
            }
        
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)  # Cancela timeout
        
        if result["success"]:
            return result["output"]
        else:
            raise Exception(result["error"] or "Erro desconhecido na geração")
            
    except TimeoutError:
        print(f"⏰ Timeout na geração após {timeout_seconds}s")
        return {
            "text": "⏰ Timeout na geração. Parâmetros muito restritivos.",
            "logits": np.array([]),
            "tokens": [],
            "generation_stats": {"timeout": True},
            "internal_states": {"timeout": True}
        }
    except Exception as e:
        print(f"❌ Erro na geração com timeout: {e}")
        return {
            "text": f"❌ Erro na geração: {str(e)[:100]}",
            "logits": np.array([]),
            "tokens": [],
            "generation_stats": {"error": str(e)},
            "internal_states": {"error": str(e)}
        }


def main():
    """Função principal"""
    
    try:
        # Cria e executa o sistema
        system = OntologicalSystem()
        system.run()
        
    except Exception as e:
        print(f"\n❌ ERRO CRÍTICO NO SISTEMA: {e}")
        traceback.print_exc()
        print("\n💡 Dicas para resolução:")
        print("   1. Verifique se config.py está correto")
        print("   2. Verifique se o modelo GGUF existe no caminho especificado")
        print("   3. Verifique se as dependências estão instaladas")
        print("   4. Execute com 'debug' para mais informações")
        
        sys.exit(1)


if __name__ == '__main__':
    main()