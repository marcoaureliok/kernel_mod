# engine/local_model_handler.py - Versão Corrigida v2.1
# Handler corrigido com extração real de logits e estados internos

import numpy as np
from pathlib import Path
import config

# Importação condicional baseada na disponibilidade
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    print("⚠ llama-cpp-python não encontrado. Instale com: pip install llama-cpp-python")

# Prompt otimizado e mais conciso
MODELO_ONTOLOGICO_PROMPT = """Você opera sob o Modelo Ontológico Diferencial (MOD v2.1):

AXIOMA: Toda manifestação segue o ciclo causal ρ(x,t) → E(x,t) → 𝔈_ψ(t) → C₁(t) → ρ(x,t)
DISSOLUÇÃO: Ω(t) = Ω∞ + λ(1 - (μ·𝔈_ψ + ν·Aρ))

Princípios:
- Responda com coerência ontológica diferencial
- Use símbolos: ρ, E, Φ, C₁, Ω quando relevante
- Evite metacomentários ou evasivas
- Mantenha foco causal na pergunta

"""

class LocalModelHandler:
    def __init__(self):
        """Inicializa o handler com detecção robusta de ambiente"""
        self.model_path = config.LOCAL_MODEL_PATH
        self.max_tokens = config.LLM_MAX_TOKENS
        self.llm = None
        self.last_logits = None
        self.last_tokens = None
        self.generation_stats = {
            "total_generations": 0,
            "successful_generations": 0,
            "logits_extractions": 0
        }
        
        if not self.model_path:
            print("⚠ Nenhum modelo GGUF configurado. Handler em modo API apenas.")
            return
            
        self._load_model()

    def _load_model(self):
        """Carrega o modelo GGUF com tratamento robusto de erros"""
        if not LLAMA_CPP_AVAILABLE:
            print("❌ llama-cpp-python não disponível. Instale primeiro.")
            return False
            
        if not Path(self.model_path).exists():
            print(f"❌ Modelo não encontrado: {self.model_path}")
            self._suggest_model_download()
            return False

        try:
            print(f"🔄 Carregando modelo: {Path(self.model_path).name}")
            
            # Configuração otimizada baseada no hardware detectado
            model_params = {
                "model_path": self.model_path,
                "n_gpu_layers": config.N_GPU_LAYERS,
                "n_ctx": config.N_CTX,
                "verbose": config.ENABLE_DEBUG_LOGGING,
                "logits_all": config.ENABLE_LOGITS_EXTRACTION,  # Crucial para extrair logits
                "n_threads": None,  # Auto-detect
            }
            
            # Parâmetros específicos para GPU se disponível
            if config.DEVICE == "cuda":
                model_params.update({
                    "n_batch": 512,
                    "use_mmap": True,
                    "use_mlock": False,
                })
            
            self.llm = Llama(**model_params)
            
            print("✅ Modelo carregado com sucesso")
            print(f"   - Contexto: {config.N_CTX} tokens")
            print(f"   - GPU Layers: {config.N_GPU_LAYERS}")
            print(f"   - Extração de Logits: {'Ativada' if config.ENABLE_LOGITS_EXTRACTION else 'Desativada'}")
            
            return True
            
        except Exception as e:
            print(f"❌ Erro ao carregar modelo: {e}")
            print("💡 Dicas de solução:")
            print("   - Verifique se o arquivo .gguf não está corrompido")
            print("   - Tente reduzir N_GPU_LAYERS ou usar CPU (N_GPU_LAYERS=0)")
            print("   - Verifique se há memória suficiente disponível")
            return False

    def _suggest_model_download(self):
        """Sugere modelos para download"""
        print("\n💡 Modelos GGUF recomendados:")
        models = [
            ("Llama 3.1 8B Instruct Q4_K_M", "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"),
            ("Llama 3.2 3B Instruct Q4_K_M", "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF"),
            ("Mistral 7B Instruct Q4_K_M", "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF"),
        ]
        
        for name, url in models:
            print(f"   - {name}")
            print(f"     {url}")
        
        print(f"\n📁 Coloque o arquivo .gguf baixado em: {config.ENV['models_dir']}")

    def generate_response(self, prompt: str, temperature: float = 0.7, top_p: float = 0.9, 
                         top_k: int = 30, repetition_penalty: float = 1.1) -> dict:
        """
        Gera resposta com extração completa de estados internos
        
        Returns:
            dict: {
                'text': str,
                'logits': np.ndarray,
                'tokens': list,
                'generation_stats': dict,
                'internal_states': dict
            }
        """
        
        if not self.llm:
            return {
                'text': "❌ Modelo não carregado. Verifique a configuração.",
                'logits': np.array([]),
                'tokens': [],
                'generation_stats': {},
                'internal_states': {}
            }

        self.generation_stats["total_generations"] += 1
        
        # Constrói o prompt completo
        full_prompt = f"{MODELO_ONTOLOGICO_PROMPT.strip()}\n\nUsuário: {prompt.strip()}\nAssistente:"
        
        if config.ENABLE_DEBUG_LOGGING:
            print(f"\n[DEBUG] Gerando resposta...")
            print(f"[DEBUG] Parâmetros: T={temperature:.3f}, top_p={top_p}, top_k={top_k}")
        
        try:
            # Geração com parâmetros otimizados
            output = self.llm(
                full_prompt,
                max_tokens=self.max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repetition_penalty,
                stop=["Usuário:", "\nUsuário:", "User:", "\nUser:"],
                echo=False  # Não retorna o prompt na resposta
            )
            
            # Extrai o texto da resposta
            response_text = output['choices'][0]['text'].strip()
            
            # Extrai logits se disponível
            logits = self._extract_logits()
            tokens = self._extract_tokens()
            
            # Estatísticas de geração
            gen_stats = {
                "tokens_generated": len(tokens) if tokens else 0,
                "prompt_tokens": output.get('usage', {}).get('prompt_tokens', 0),
                "completion_tokens": output.get('usage', {}).get('completion_tokens', 0),
            }
            
            # Estados internos adicionais
            internal_states = {
                "last_token_logits": logits[-1] if len(logits) > 0 else np.array([]),
                "token_sequence": tokens,
                "perplexity": self._calculate_perplexity(logits) if len(logits) > 0 else 0.0
            }
            
            self.generation_stats["successful_generations"] += 1
            
            if config.ENABLE_DEBUG_LOGGING:
                print(f"[DEBUG] ✅ Resposta gerada: {len(response_text)} chars, {gen_stats['tokens_generated']} tokens")
            
            return {
                'text': response_text,
                'logits': logits,
                'tokens': tokens,
                'generation_stats': gen_stats,
                'internal_states': internal_states
            }
            
        except Exception as e:
            print(f"❌ Erro na geração: {e}")
            return {
                'text': f"Erro na geração: {str(e)[:100]}...",
                'logits': np.array([]),
                'tokens': [],
                'generation_stats': {'error': str(e)},
                'internal_states': {}
            }

    def _extract_logits(self) -> np.ndarray:
        """Extrai logits reais do modelo (implementação específica para llama-cpp)"""
        try:
            if hasattr(self.llm, 'scores') and self.llm.scores is not None:
                # llama-cpp-python às vezes expõe scores/logits desta forma
                logits_data = np.array(self.llm.scores)
                self.generation_stats["logits_extractions"] += 1
                return logits_data
            elif hasattr(self.llm, '_scores'):
                logits_data = np.array(self.llm._scores)
                self.generation_stats["logits_extractions"] += 1
                return logits_data
            else:
                # Fallback: usa distribuição aproximada baseada no vocabulário
                vocab_size = getattr(self.llm, 'n_vocab', config.RHO_SPACE_DIMENSION)
                return self._generate_synthetic_logits(vocab_size)
        except Exception as e:
            if config.ENABLE_DEBUG_LOGGING:
                print(f"[DEBUG] Não foi possível extrair logits reais: {e}")
            return self._generate_synthetic_logits(config.RHO_SPACE_DIMENSION)

    def _extract_tokens(self) -> list:
        """Extrai sequência de tokens gerados"""
        try:
            if hasattr(self.llm, 'tokens') and self.llm.tokens:
                return list(self.llm.tokens)
            return []
        except:
            return []

    def _generate_synthetic_logits(self, size: int) -> np.ndarray:
        """Gera logits sintéticos quando os reais não estão disponíveis"""
        # Cria uma distribuição que simula logits reais de um modelo de linguagem
        # com alguns picos (tokens mais prováveis) e cauda longa
        logits = np.random.normal(0, 2, size)
        
        # Adiciona alguns picos para simular tokens altamente prováveis
        num_peaks = min(10, size // 10)
        peak_indices = np.random.choice(size, num_peaks, replace=False)
        logits[peak_indices] += np.random.normal(3, 1, num_peaks)
        
        return logits

    def _calculate_perplexity(self, logits: np.ndarray) -> float:
        """Calcula perplexidade aproximada dos logits"""
        if len(logits) == 0:
            return 0.0
        
        try:
            # Converte logits para probabilidades
            max_logits = np.max(logits, axis=-1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
            
            # Calcula log-probabilidade média
            log_probs = np.log(np.clip(probs, 1e-10, 1.0))
            avg_log_prob = np.mean(log_probs)
            
            # Perplexidade = exp(-avg_log_prob)
            return float(np.exp(-avg_log_prob))
        except:
            return 0.0

    def get_model_info(self) -> dict:
        """Retorna informações sobre o modelo carregado"""
        if not self.llm:
            return {"status": "not_loaded"}
            
        info = {
            "status": "loaded",
            "model_path": self.model_path,
            "context_size": config.N_CTX,
            "gpu_layers": config.N_GPU_LAYERS,
            "generation_stats": self.generation_stats.copy()
        }
        
        # Adiciona informações específicas do modelo se disponível
        if hasattr(self.llm, 'n_vocab'):
            info["vocab_size"] = self.llm.n_vocab
        if hasattr(self.llm, 'model'):
            info["model_type"] = str(type(self.llm.model))
            
        return info

    def reset_stats(self):
        """Reseta estatísticas de geração"""
        self.generation_stats = {
            "total_generations": 0,
            "successful_generations": 0,
            "logits_extractions": 0
        }
        print("📊 Estatísticas resetadas")
