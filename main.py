# main.py (Versão Corrigida com Ciclo de Refinamento)

import config
from kernel.kernel import OntologicalKernel
from engine.local_model_handler import LocalModelHandler
import numpy as np

def apply_regulation(regulation_params: dict) -> dict:
    # Esta função permanece a mesma
    omega = regulation_params.get('omega', 0.1)
    c1 = regulation_params.get('c1', 0.5)
    temperature = 0.7 + (omega * 1.2) - (c1 * 0.3)
    final_temperature = max(0.1, min(temperature, 1.8))
    print("\n   >>> Ação Regulatória do Kernel <<<")
    print(f"   Dissolução (Ω)={omega:.3f}, Intensidade (C₁)={c1:.3f}")
    print(f"   Parâmetro de Geração Ajustado: Temperatura = {final_temperature:.2f}")
    print("   ----------------------------------")
    return {"temperature": final_temperature, "top_p": .15, "top_k": 30, "repetition_penalty": 1.1}

def create_corrective_prompt(original_prompt: str, diagnosis: dict) -> str:
    """Cria um novo prompt para corrigir a IA com base no diagnóstico."""
    # Lógica simples para começar:
    if diagnosis.get('reflexive_degeneration', 0) > config.REFINEMENT_DEGENERATION_THRESHOLD:
        return f"Sua resposta anterior foi evasiva ou um metacomentário. Responda diretamente e de forma substancial à pergunta original. Pergunta Original: '{original_prompt}'"
    return original_prompt # Retorna o original se nenhum problema grave for detectado

def main():
    print("================================================================")
    print("=== KERNEL ONTOLÓGICO v2.1 (COM CICLO DE REFINAMENTO ATIVO) ===")
    print("================================================================")
    
    model_handler = LocalModelHandler()
    kernel = OntologicalKernel(config)
    
    print("\nDigite 'sair' para terminar a sessão.")
    regulation_params = {"temperature": 0.7, "top_p": .15, "top_k": 30, "repetition_penalty": 1.1}

    while True:
        user_prompt = input("\nVocê: ")
        if user_prompt.lower() == 'sair':
            break

        # --- Ciclo Pré-Geração ---
        print("\n--- Ciclo Pré-Geração (Análise do Prompt) ---")
        prompt_state = {"text": user_prompt, "logits": np.array([ord(c) for c in user_prompt])}
        prompt_control_output = kernel.run_cycle(prompt_state)
        current_regulation_params = apply_regulation(prompt_control_output['regulation'])
        
        # --- Geração Inicial e Ciclo de Refinamento ---
        attempts = 0
        is_response_satisfactory = False
        ai_response = ""

        while attempts <= config.MAX_REFINEMENT_ATTEMPTS and not is_response_satisfactory:
            if attempts == 0:
                print("\nIA está pensando... (Tentativa 1)")
                prompt_para_ia = user_prompt
            else:
                print(f"\nIA está pensando... (Tentativa de Refinamento {attempts})")

            # Gera a resposta
            ai_response = model_handler.generate_response(prompt_para_ia, **current_regulation_params)
            
            # --- Ciclo Pós-Geração (Vigilância) ---
            print("--- Ciclo Pós-Geração (Analisando Resposta) ---")
            response_state = {"text": ai_response, "logits": np.array([ord(c) for c in ai_response])}
            post_control_output = kernel.run_cycle(response_state)
            
            # Verifica se a resposta é boa o suficiente
            diagnosis = post_control_output['diagnostics']
            if diagnosis.get('reflexive_degeneration', 0) < config.REFINEMENT_DEGENERATION_THRESHOLD or not config.ENABLE_REFINEMENT_CYCLE:
                is_response_satisfactory = True
                print("   >>> Diagnóstico: Resposta Aprovada pelo Kernel.")
                # Calcula os parâmetros para a PRÓXIMA rodada
                regulation_params = apply_regulation(post_control_output['regulation'])
            else:
                print("   >>> Diagnóstico: Resposta Reprovada. Alta Degeneração Reflexiva. Iniciando Refinamento...")
                attempts += 1
                # Cria um novo prompt corretivo para a próxima tentativa
                prompt_para_ia = create_corrective_prompt(user_prompt, diagnosis)
                # Mantém os mesmos parâmetros de temperatura para a nova tentativa
        
        # Exibe a resposta final (a que foi aprovada ou a última tentativa)
        print(f"\nIA: {ai_response}")

    print("\n=========================\n=== SESSÃO ENCERRADA ===\n=========================")

if __name__ == '__main__':
    main()