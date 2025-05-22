# Módulo para gerenciar a memória persistente individual dos usuários usando mem0

from mem0ai import Memory # Corrigido para usar o pacote instalado
import os

# Diretório para armazenar as memórias dos usuários
MEMORY_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'memories', 'users'))

# Garante que o diretório de memórias exista
if not os.path.exists(MEMORY_DIR):
    os.makedirs(MEMORY_DIR)

def get_user_memory(user_id: str) -> Memory:
    """Retorna uma instância de memória para um usuário específico."""
    # Define um caminho específico para a memória do usuário
    # No futuro, podemos usar um config mais robusto ou um DB
    user_memory_path = os.path.join(MEMORY_DIR, f"memory_{user_id}")
    
    # Inicializa o mem0 com um diretório persistente
    # Veja a documentação do mem0 para mais opções de configuração
    memory = Memory.from_config({
        "vector_store": {
            "provider": "chroma",
            "config": {
                "host": None, # Usará ChromaDB localmente
                "path": user_memory_path
            }
        },
        "llm": {
            "provider": "openai", # Placeholder, pode ser configurado depois
            "config": {
                "model": "gpt-3.5-turbo"
            }
        }
    })
    return memory

def save_memory(user_id: str, text: str):
    """Salva uma informação na memória do usuário."""
    memory = get_user_memory(user_id)
    memory.add(text)
    print(f"Memória salva para o usuário {user_id}.")

def load_memory(user_id: str, query: str):
    """Busca informações na memória do usuário."""
    memory = get_user_memory(user_id)
    results = memory.search(query)
    print(f"Resultados da busca na memória para {user_id}: {results}")
    return results

if __name__ == "__main__":
    # Exemplo de uso
    test_user = "user_123"
    print(f"Inicializando memória para {test_user} em {MEMORY_DIR}")
    save_memory(test_user, "O usuário prefere respostas curtas.")
    save_memory(test_user, "O nome do gato do usuário é Fígaro.")
    load_memory(test_user, "Qual o nome do gato do usuário?")
    load_memory(test_user, "Qual a preferência de resposta do usuário?")
    print("\nMódulo de memória de usuário (mem0) testado.")