# Script para processar e indexar conhecimento usando Docling e FAISS

import os
from docling import process_documents # Supondo função de processamento Docling
from faiss import faiss_retriever
import numpy as np

# Diretórios
SOURCES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'knowledge', 'sources'))
INDEX_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'knowledge', 'faiss_index'))
INDEX_FILE = os.path.join(INDEX_DIR, 'knowledge.index')

# Função para processar e indexar conhecimento
def process_and_index_knowledge():
    print("Iniciando processamento de conhecimento com Docling...")
    # 1. Ler arquivos do diretório de fontes
    sources = []
    for fname in os.listdir(SOURCES_DIR):
        fpath = os.path.join(SOURCES_DIR, fname)
        if os.path.isfile(fpath):
            with open(fpath, 'r', encoding='utf-8') as f:
                sources.append(f.read())
    if not sources:
        print("Nenhuma fonte encontrada em knowledge/sources.")
        return
    # 2. Processar conhecimento com Docling (gera embeddings e metadados)
    print(f"Processando {len(sources)} documentos...")
    embeddings, metadados = process_documents(sources) # embeddings: np.ndarray, metadados: list
    # 2.1. Salvar metadados em JSON
    metadados_file = os.path.join(INDEX_DIR, 'knowledge_metadata.json')
    import json
    with open(metadados_file, 'w', encoding='utf-8') as mf:
        json.dump(metadados, mf, ensure_ascii=False, indent=2)
    print(f"Metadados salvos em {metadados_file}")
    # 3. Indexar vetores no FAISS
    print("Indexando vetores no FAISS...")
    if np.array(embeddings).shape[1] != faiss_retriever.DIMENSION:
        print(f"[AVISO] Dimensão dos embeddings ({np.array(embeddings).shape[1]}) difere da configuração do índice FAISS ({faiss_retriever.DIMENSION})!")
    index = faiss_retriever.faiss.IndexFlatL2(faiss_retriever.DIMENSION)
    index.add(np.array(embeddings).astype('float32'))
    faiss.write_index(index, INDEX_FILE)
    print(f"Indexação concluída. Índice salvo em {INDEX_FILE}")
    # 4. Log simples
    print(f"{len(embeddings)} vetores indexados.")

if __name__ == "__main__":
    print("Processamento e indexação de conhecimento iniciado.")
    process_and_index_knowledge()