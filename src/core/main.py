# Entry point do backend Python

from mem0 import MemoryClient # Importa o MemoryClient
from src.data_persistence.faiss import faiss_retriever # Importa o módulo FAISS
from src.core.langgraph.graph_builder import run_graph # Importa a função de execução do grafo
import os # Para acessar variáveis de ambiente
import numpy as np # Necessário para criar vetores de teste

def main():
    print("Backend Hubblet AI iniciado.")
    # Exemplo de uso da memória
    test_user = "backend_main_test"
    print(f"Testando memória para {test_user}")
    mem0_api_key = os.environ.get("MEM0_API_KEY")
    if not mem0_api_key:
        print("AVISO: MEM0_API_KEY não configurada. Testes de memória podem falhar.")
        mem0_client = MemoryClient() # Inicialização padrão se a chave não estiver presente
    else:
        mem0_client = MemoryClient(api_key=mem0_api_key)

    # Adicionando uma memória de teste
    mem0_client.add("Esta é uma mensagem de teste do main.py.", user_id=test_user, agent_id="main_agent")
    print("Memória de teste adicionada.")

    # Buscando na memória
    results = mem0_client.search("Qual a mensagem de teste?", user_id=test_user, agent_id="main_agent")
    print(f"Resultado da busca na memória: {results}")

    # Exemplo de uso da busca de conhecimento (FAISS)
    print("\nTestando busca na base de conhecimento (FAISS)...")
    # Cria um vetor de consulta simulado (a dimensão deve corresponder à definida em faiss_retriever)
    # No futuro, este vetor viria de um modelo de embedding
    try:
        mock_dimension = faiss_retriever.DIMENSION
        mock_query_vector = np.random.rand(mock_dimension).astype('float32')
        distances, indices = faiss_retriever.search_knowledge(mock_query_vector, k=3)
        print(f"Resultado da busca no FAISS (simulado): Distâncias={distances}, Índices={indices}")
    except Exception as e:
        print(f"Erro ao testar busca FAISS: {e}")

    # Exemplo de execução do LangGraph
    print("\nTestando execução do LangGraph...")
    test_graph_input = "Qual a relação entre gatos e felinos?"
    try:
        graph_result = run_graph(test_graph_input)
        print("\nResultado final do grafo LangGraph:")
        print(graph_result)
    except Exception as e:
        print(f"Erro ao executar o grafo LangGraph: {e}")

    # Aqui virá a lógica principal: API, etc.

if __name__ == "__main__":
    main()