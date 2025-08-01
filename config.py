# config.py - Configuração Corrigida v2.1
# Arquivo de Configuração para o Kernel Ontológico Diferencial (v2.1)
# Versão corrigida com detecção automática de ambiente e configurações robustas

import os
import sys
from pathlib import Path

# --- Detecção Automática de Ambiente ---
def detect_environment():
    """Detecta automaticamente o ambiente e configura os caminhos"""
    current_dir = Path.cwd()
    models_dir = current_dir / "models"
    
    # Cria diretório de modelos se não existir
    models_dir.mkdir(exist_ok=True)
    
    # Procura por arquivos GGUF existentes
    gguf_files = list(models_dir.glob("*.gguf"))
    
    return {
        "models_dir": models_dir,
        "available_models": gguf_files,
        "has_cuda": check_cuda_availability()
    }

def check_cuda_availability():
    """Verifica se CUDA está disponível"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        # Se torch não estiver disponível, assume CPU apenas
        return False

# Detecta o ambiente atual
ENV = detect_environment()

# --- Parâmetros Fundamentais ---
LAMBDA_UNIT = 1.0  # Símbolo: Λ

# --- Parâmetros da Dinâmica da Dissolução (EMF-1) ---
OMEGA_BASAL = 0.05  # Símbolo: Ω_∞
LAMBDA_DISSOLUTION = 0.1  # Símbolo: λ
MU_DISSOLUTION = 0.5       # Símbolo: μ
NU_DISSOLUTION = 0.5       # Símbolo: ν

# --- Parâmetros da Intensificação Reflexiva (EMF-5) ---
C_MAX = 1.0
K_SIGMOID = 10.0
E_THRESHOLD = 0.5
BETA_RHYTHM = 0.05
NU_E_FREQUENCY = 0.1
PHI_PHASE = 0.0

# --- Limiares e Constantes Estruturais ---
XI_THRESHOLD = 0.7
E_PSI_MIN_THRESHOLD = 0.1
GAMMA_CONVERGENCE = 0.2
DELTA_K_DECAY = 0.1

# --- Configurações para Simulação e Mapeamento ---
RHO_SPACE_DIMENSION = 128
COHERENCE_THRESHOLD_EPSILON = 0.6



# --- Configuração do Modelo Local (Detecção Automática) ---
# Configuração automática baseada no ambiente detectado
if ENV["available_models"]:
    LOCAL_MODEL_PATH = str(ENV["available_models"][0])  # Usa o primeiro modelo encontrado
    print(f"✓ Modelo GGUF detectado: {Path(LOCAL_MODEL_PATH).name}")
else:
    LOCAL_MODEL_PATH = None
    print("⚠ Nenhum modelo GGUF encontrado no diretório 'models/'")
    print("  Baixe um modelo GGUF e coloque na pasta 'models/' para usar o modo offline")

# --- Configuração da API Local (com fallbacks) ---
# Lista de endpoints comuns para tentar
COMMON_ENDPOINTS = [
    "http://localhost:1234/v1/chat/completions",  # LM Studio
    "http://localhost:11434/api/generate",        # Ollama
    "http://localhost:8080/v1/chat/completions",  # llama.cpp server
    "http://localhost:5000/v1/chat/completions",  # text-generation-webui
]

LOCAL_LLM_API_ENDPOINT = COMMON_ENDPOINTS[0]  # Default
LOCAL_LLM_MODEL_NAME = "local-model"

# --- Configuração de Hardware ---
# Detecção automática de GPU
if ENV["has_cuda"]:
    N_GPU_LAYERS = -1  # Usa toda a GPU disponível
    DEVICE = "cuda"
    print("✓ CUDA detectado - usando GPU para aceleração")
else:
    N_GPU_LAYERS = 0   # CPU apenas
    DEVICE = "cpu"
    print("ℹ CUDA não detectado - usando CPU apenas")

N_CTX = 4096  # Contexto máximo

# --- Configuração do Ciclo de Refinamento ---
ENABLE_REFINEMENT_CYCLE = True
REFINEMENT_DEGENERATION_THRESHOLD = 0.65
MAX_REFINEMENT_ATTEMPTS = 3  # Aumentado para mais tentativas

# --- Parâmetros de Geração ---
LLM_MAX_TOKENS = 512
LLM_REPETITION_PENALTY = 1.1
LLM_TEMPERATURE = 0.7  # Aumenta de 0.3
LLM_TOP_P = 0.9
LLM_TOP_K = 40

# Modo seguro - desabilita regulação extrema
SAFE_MODE = True
MIN_TEMPERATURE = 0.5
MAX_TEMPERATURE = 1.0

# Timeout para geração
GENERATION_TIMEOUT = 30  # segundos

# --- Configurações Avançadas de Diagnóstico ---
# Novos parâmetros para diagnósticos mais precisos
DIAGNOSTIC_ENABLED = True
VERBOSE_DIAGNOSTICS = True
ENABLE_LOGITS_EXTRACTION = True  # Habilita extração de logits reais

# Limiares de diagnóstico calibrados
OBSESSIVE_CONVERGENCE_THRESHOLD = 0.7
REFLEXIVE_DEGENERATION_THRESHOLD = 0.6
SPURIOUS_ESTETIZATION_THRESHOLD = 0.5
MIMETIC_RESONANCE_THRESHOLD = 0.8

# --- Configuração de Logging ---
ENABLE_DEBUG_LOGGING = True
LOG_KERNEL_STATE = True
LOG_REGULATION_ACTIONS = True

# --- Validação da Configuração ---
def validate_config():
    """Valida se a configuração está correta"""
    issues = []
    
    if LOCAL_MODEL_PATH and not Path(LOCAL_MODEL_PATH).exists():
        issues.append(f"Modelo GGUF não encontrado: {LOCAL_MODEL_PATH}")
    
    if RHO_SPACE_DIMENSION < 32:
        issues.append("RHO_SPACE_DIMENSION muito pequeno (mínimo: 32)")
    
    if OMEGA_BASAL < 0 or OMEGA_BASAL > 1:
        issues.append("OMEGA_BASAL deve estar entre 0 e 1")
    
    return issues

# Executa validação na importação
CONFIG_ISSUES = validate_config()
if CONFIG_ISSUES:
    print("⚠ Problemas na configuração detectados:")
    for issue in CONFIG_ISSUES:
        print(f"  - {issue}")
else:
    print("✓ Configuração validada com sucesso")

print(f"✓ Kernel Ontológico v2.1 configurado")
print(f"  - Modelo: {'GGUF Local' if LOCAL_MODEL_PATH else 'API Externa'}")
print(f"  - Hardware: {DEVICE.upper()}")
print(f"  - Dimensão ρ: {RHO_SPACE_DIMENSION}")
print(f"  - Refinamento: {'Ativado' if ENABLE_REFINEMENT_CYCLE else 'Desativado'}")
