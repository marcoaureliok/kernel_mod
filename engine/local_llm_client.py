# local_llm_client.py
# Cliente para interagir com um modelo de linguagem local (LLM).
# Este módulo abstrai a comunicação com a API da IA.

import requests
import json
import config

class LocalLLMClient:
    def __init__(self):
        """
        Inicializa o cliente com as configurações do arquivo config.py.
        """
        self.api_url = config.LOCAL_LLM_API_ENDPOINT
        self.model_name = config.LOCAL_LLM_MODEL_NAME
        print(f"Cliente de IA Local configurado para o endpoint: {self.api_url}")

    def generate_response(self, prompt: str, temperature: float = 0.7) -> str:
        """
        Envia um prompt para a IA local e retorna a resposta textual.

        Args:
            prompt (str): O texto de entrada para a IA.
            temperature (float): O parâmetro de geração que será controlado
                                 pelo nosso kernel (via regulação de Ω).

        Returns:
            str: A resposta textual gerada pela IA.
        """
        headers = {"Content-Type": "application/json"}
        # O formato do corpo da requisição pode variar dependendo da sua API.
        # Este é um exemplo comum para APIs compatíveis com a OpenAI (como o LM Studio).
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "stream": False
        }

        try:
            response = requests.post(self.api_url, headers=headers, data=json.dumps(data))
            response.raise_for_status() # Lança um erro para respostas HTTP 4xx/5xx

            response_json = response.json()
            # A extração do conteúdo da resposta também pode variar.
            content = response_json['choices'][0]['message']['content']
            return content.strip()

        except requests.exceptions.RequestException as e:
            print(f"ERRO: Não foi possível conectar à API da IA local em {self.api_url}.")
            print(f"Detalhe do erro: {e}")
            return "Erro: A comunicação com o modelo local falhou."
        except (KeyError, IndexError) as e:
            print(f"ERRO: A resposta da API não estava no formato esperado.")
            print(f"Detalhe do erro: {e}")
            print(f"Resposta recebida: {response.text}")
            return "Erro: Formato de resposta inesperado do modelo."

print("Arquivo 'local_llm_client.py' criado. Pronto para se comunicar com a IA.")