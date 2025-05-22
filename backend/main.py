# Entry point do backend Python

from mem0.user_memory import save_memory, load_memory # Importa funções específicas
from faiss import faiss_retriever # Importa o módulo FAISS
from langgraph.graph_builder import run_graph # Importa a função de execução do grafo
import numpy as np # Necessário para criar vetores de teste

def main():
    print("Backend Hubblet AI iniciado.")
    # Exemplo de uso da memória
    test_user = "backend_main_test"
    print(f"Testando memória para {test_user}")
    save_memory(test_user, "Esta é uma mensagem de teste do main.py.") # Chama a função diretamente
    results = load_memory(test_user, "Qual a mensagem de teste?") # Chama a função diretamente
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