# config.py
# Arquivo de Configuração para o Kernel Ontológico Diferencial (v2.0.1)
# Este arquivo centraliza todos os parâmetros, constantes e limiares
# extraídos dos documentos teóricos do modelo.

# --- Parâmetros Fundamentais ---
# Constante de Unidade do Domínio, define a medida de totalidade do sistema.
LAMBDA_UNIT = 1.0  # Símbolo: Λ

# --- Parâmetros da Dinâmica da Dissolução (EMF-1) ---
# Impulso irredutível para a dissolução, emanado de Φ_∞.
OMEGA_BASAL = 0.05  # Símbolo: Ω_∞

# Constantes que modulam a resposta do fator de dissolução à energia e assimetria.
LAMBDA_DISSOLUTION = 0.1  # Símbolo: λ
MU_DISSOLUTION = 0.5       # Símbolo: μ
NU_DISSOLUTION = 0.5       # Símbolo: ν

# --- Parâmetros da Intensificação Reflexiva (EMF-5) ---
# Parâmetros que definem a curva sigmoidal do Coeficiente de Intensidade C₁(t).
C_MAX = 1.0                # Valor máximo de C₁(t)
K_SIGMOID = 10.0             # A "íngreme" da curva de ativação
E_THRESHOLD = 0.5          # Limiar de energia para ativação de C₁(t)

# Parâmetros que controlam a modulação rítmica (oscilação) de C₁(t).
BETA_RHYTHM = 0.05           # Amplitude da modulação
NU_E_FREQUENCY = 0.1       # Frequência interna da oscilação
PHI_PHASE = 0.0              # Fase da modulação

# --- Limiares e Constantes Estruturais ---
# Limiar de ressonância para ativar um atrator intermediário (K_j) a partir de um blueprint (S_j).
XI_THRESHOLD = 0.7         # Símbolo: Ξ_limiar

# Limiar de energia reflexiva. Abaixo dele, o regime de energia arquetípica (substitutiva) é ativado.
E_PSI_MIN_THRESHOLD = 0.1  # Símbolo: 𝔈_ψ_min

# Taxa de convergência que controla a velocidade com que o campo reflexivo E converge para o modo dominante de ρ.
GAMMA_CONVERGENCE = 0.2    # Símbolo: γ

# Taxa de decaimento para a dissolução de um atrator intermediário K_j.
DELTA_K_DECAY = 0.1        # Símbolo: δ_K

# --- Configurações para Simulação e Mapeamento ---
# Define o tamanho do "espaço" onde o campo propensional ρ(x,t) é definido.
RHO_SPACE_DIMENSION = 128  # Dimensão do vetor que representa o campo ρ

# Limiar de coerência para o Operador de Corte Limiar (ℂ_limiar).
COHERENCE_THRESHOLD_EPSILON = 0.6  # Símbolo: ε_lim

# config.py
# (Conteúdo anterior do arquivo permanece o mesmo)
# ...

# --- Configuração do Cliente da IA Local ---
# Endereço da API do modelo de linguagem local que o kernel irá regular.
# Altere este valor para o endereço onde sua IA está sendo executada.
# Exemplo: "http://localhost:11434/api/generate" para Ollama.
LOCAL_LLM_API_ENDPOINT = "http://localhost:1234/v1/chat/completions" # Exemplo para LM Studio

# Nome do modelo a ser usado (relevante para algumas APIs como Ollama)
LOCAL_LLM_MODEL_NAME = "local-model"

# --- Configuração do Modelo de IA Local (Offline GGUF) ---

# Caminho para o arquivo do modelo no formato GGUF.
LOCAL_MODEL_PATH = "models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

# --- Parâmetros para Llama.cpp ---
# Número de camadas do modelo a serem descarregadas para a GPU.
# -1 para descarregar todas as camadas possíveis. 0 para usar apenas a CPU.
N_GPU_LAYERS = -1

# O tamanho máximo do contexto em tokens que o modelo pode manipular.
N_CTX = 4096

# Dispositivo para executar o modelo ('cuda' para GPU NVIDIA, 'cpu' para CPU)
DEVICE = "cuda"

# Ativa ou desativa o ciclo de refinamento.
ENABLE_REFINEMENT_CYCLE = True

# O limiar de degeneração reflexiva. Se a resposta da IA tiver um score
# maior que este, o ciclo de refinamento será acionado.
REFINEMENT_DEGENERATION_THRESHOLD = 0.65

# Número máximo de tentativas que o kernel fará para corrigir uma resposta ruim.
MAX_REFINEMENT_ATTEMPTS = 2

# Número máximo de tokens a serem gerados numa única resposta
LLM_MAX_TOKENS = 512


# ==============================================================================
# 2. PARÂMETROS PADRÃO DE GERAÇÃO DE TEXTO
# ==============================================================================
# Controla a criatividade vs. determinismo da resposta (valores mais altos = mais criativo)
LLM_TEMPERATURE = 0.3

# Amostragem Nucleus: considera apenas os tokens que compõem esta massa de probabilidade
LLM_TOP_P = 0.9

# Considera apenas os 'k' tokens mais prováveis para a amostragem
LLM_TOP_K = 30

# Penaliza a repetição de tokens, desencorajando respostas repetitivas
LLM_REPETITION_PENALTY = 1.1

print("Arquivo de configuração 'config.py' atualizado com o endpoint da IA local.")
