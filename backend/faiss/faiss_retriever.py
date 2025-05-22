# Módulo para buscar informações no índice FAISS

import faiss
import numpy as np
import os

# Diretório onde o índice FAISS será armazenado
INDEX_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'knowledge', 'faiss_index'))
INDEX_FILE = os.path.join(INDEX_DIR, 'knowledge.index')

# Garante que o diretório do índice exista
if not os.path.exists(INDEX_DIR):
    os.makedirs(INDEX_DIR)

DIMENSION = 128 # Dimensão dos vetores (deve ser igual ao usado no process_knowledge.py)

_cached_index = None

def load_faiss_index():
    """Carrega o índice FAISS do disco (real)."""
    global _cached_index
    if _cached_index is not None:
        return _cached_index
    if os.path.exists(INDEX_FILE):
        print(f"Carregando índice FAISS de {INDEX_FILE}")
        index = faiss.read_index(INDEX_FILE)
        print("Índice FAISS carregado com sucesso.")
    else:
        print("Arquivo de índice FAISS não encontrado. Criando índice vazio.")
        index = faiss.IndexFlatL2(DIMENSION)
    _cached_index = index
    return index

def search_knowledge(query_vector: np.ndarray, k: int = 5):
    """Busca os k vizinhos mais próximos no índice FAISS real."""
    index = load_faiss_index()
    if index.ntotal == 0:
        print("Índice FAISS está vazio. Nenhuma busca realizada.")
        return [], []
    print(f"Buscando {k} vizinhos para o vetor de consulta...")
    distances, indices = index.search(np.array([query_vector]).astype('float32'), k)
    print(f"Busca no FAISS concluída. Distâncias: {distances}, Índices: {indices}")
    return distances[0].tolist(), indices[0].tolist()

if __name__ == "__main__":
    print("Testando o módulo FAISS Retriever...")
    mock_query_vector = np.random.rand(DIMENSION).astype('float32')
    search_knowledge(mock_query_vector)
    print("Módulo FAISS Retriever testado.")