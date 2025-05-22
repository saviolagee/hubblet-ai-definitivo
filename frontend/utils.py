import streamlit as st
import os
import json
import faiss
import numpy as np
from mem0 import MemoryClient
from typing import List, Dict
import tempfile
from openai import OpenAI # Adicionado para gerar_embeddings
import glob

# Funções utilitárias para o frontend Hubblet AI

def inicializar_faiss(dim: int = 1536) -> faiss.Index:
    """Inicializa um índice FAISS simples em memória."""
    return faiss.IndexFlatL2(dim)

def gerar_embeddings(textos: List[str], openai_api_key: str) -> List[np.ndarray]:
    """Gera embeddings para uma lista de textos usando a API da OpenAI."""
    if not openai_api_key:
        st.error("Chave da API OpenAI (OPENAI_API_KEY) não fornecida. Embeddings não podem ser gerados.")
        return []
    
    client = OpenAI(api_key=openai_api_key)
    embeddings = []
    for i, chunk in enumerate(textos):
        if not chunk.strip(): # Pular chunks vazios ou apenas com espaços
            # st.info(f"Trecho {i+1}/{len(textos)} está vazio e será ignorado.")
            continue
        try:
            resp = client.embeddings.create(input=chunk, model="text-embedding-ada-002")
            embeddings.append(np.array(resp.data[0].embedding, dtype=np.float32))
        except Exception as e:
            st.error(f"Erro ao gerar embedding para o trecho {i+1}/{len(textos)}: {e}. Este trecho será ignorado.")
    return embeddings

def processar_arquivos(arquivos: List[st.runtime.uploaded_file_manager.UploadedFile], openai_api_key: str) -> tuple[List[str], List[np.ndarray], List[str]]:
    """Processa arquivos enviados, extrai texto, gera chunks e embeddings."""
    doc_chunks_total = []
    nomes_arquivos_processados = []
    tamanho_max_por_arquivo = 2 * 1024 * 1024  # 2MB por arquivo

    if not arquivos:
        return [], [], []

    for arq in arquivos:
        nome = arq.name
        if arq.size == 0:
            st.warning(f"Arquivo '{nome}' está vazio e será ignorado.")
            continue
        if arq.size > tamanho_max_por_arquivo:
            st.warning(f"Arquivo '{nome}' ({arq.size / (1024*1024):.2f}MB) excede o limite de 2MB e será ignorado.")
            continue
        
        tmp_path = ""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{os.path.basename(nome)}", mode='wb') as tmp:
                tmp.write(arq.getvalue()) 
                tmp_path = tmp.name
            
            texto_arquivo = ""
            try:
                with open(tmp_path, "r", encoding="utf-8") as f:
                    texto_arquivo = f.read()
            except UnicodeDecodeError:
                try:
                    with open(tmp_path, "r", encoding="latin-1") as f:
                        texto_arquivo = f.read()
                    st.info(f"Arquivo '{nome}' lido com encoding 'latin-1' após falha com 'utf-8'.")
                except Exception as e_read:
                    st.error(f"Erro ao ler o arquivo '{nome}' com diferentes encodings: {e_read}")
                    if os.path.exists(tmp_path): os.unlink(tmp_path)
                    continue 
            except Exception as e_open:
                 st.error(f"Erro ao abrir/ler o arquivo temporário para '{nome}': {e_open}")
                 if os.path.exists(tmp_path): os.unlink(tmp_path)
                 continue

            if texto_arquivo.strip():
                chunk_size = 1500  
                chunks_do_arquivo = [texto_arquivo[i:i+chunk_size] for i in range(0, len(texto_arquivo), chunk_size)]
                doc_chunks_total.extend(chunks_do_arquivo)
                nomes_arquivos_processados.append(nome)
            else:
                st.warning(f"Arquivo '{nome}' não contém texto extraível ou está vazio após a leitura.")

            if os.path.exists(tmp_path): os.unlink(tmp_path)
        except Exception as e_outer:
            st.error(f"Erro geral ao processar o arquivo '{nome}': {e_outer}")
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    if not doc_chunks_total:
        return [], [], nomes_arquivos_processados

    embeddings_gerados = []
    if openai_api_key and doc_chunks_total:
        with st.spinner(f"Gerando embeddings para {len(doc_chunks_total)} trechos de {len(nomes_arquivos_processados)} arquivo(s)..."):
            embeddings_gerados = gerar_embeddings(doc_chunks_total, openai_api_key)
            if embeddings_gerados:
                 st.success(f"{len(embeddings_gerados)} embeddings gerados.")
            elif doc_chunks_total: # Se havia chunks mas não gerou embeddings
                 st.warning("Falha ao gerar embeddings para alguns ou todos os trechos, embora houvesse conteúdo.")
    elif not openai_api_key and doc_chunks_total:
        st.warning("OPENAI_API_KEY não fornecida. Embeddings não foram gerados para os documentos processados.")

    return doc_chunks_total, embeddings_gerados, nomes_arquivos_processados

def inicializar_mem0(username: str, assistente_nome: str, api_key: str) -> MemoryClient | None:
    """Inicializa o MemoryClient do mem0 para um assistente específico."""
    if not api_key:
        st.error("Chave da API Mem0 (MEM0_API_KEY) não encontrada ou não fornecida.")
        return None
    try:
        # Usar um user_id único para cada combinação de usuário e assistente
        safe_username = username.replace(' ', '_').lower().strip()
        safe_assistente_nome = assistente_nome.replace(' ', '_').lower().strip()
        user_id_para_mem0 = f"user_{safe_username}_asst_{safe_assistente_nome}"
        return MemoryClient(api_key=api_key, user_id=user_id_para_mem0)
    except Exception as e:
        st.error(f"Erro ao inicializar Mem0 Client para '{assistente_nome}' (User ID: {user_id_para_mem0 if 'user_id_para_mem0' in locals() else 'N/A'}): {e}")
        return None

def carregar_ou_inicializar_dados_assistente(username: str, nome_assistente: str, openai_api_key: str, mem0_api_key: str):
    """Carrega dados de um assistente existente ou inicializa o estado para um novo/selecionado."""
    st.session_state["chat_history"] = []
    st.session_state["uploaded_files"] = [] # Lista de nomes de arquivos, não os objetos UploadedFile
    st.session_state["doc_chunks"] = []
    st.session_state["faiss_index"] = inicializar_faiss()
    st.session_state["instrucoes_finais"] = None
    st.session_state["mem0_instance"] = None
    st.session_state["loading_ia"] = False
    st.session_state["assistente_config"] = {"nome": nome_assistente} # Garante que o nome está na config

    if nome_assistente == "Nenhum Assistente Salvo" or nome_assistente == "Nenhum" or not nome_assistente.strip():
        st.info("Nenhum assistente específico para carregar. Estado inicializado para um novo assistente ou modo padrão.")
        return

    # Define o diretório 'assistentes_salvos' dentro do diretório 'frontend'
    # Assume que utils.py está em frontend/, então os.getcwd() pode não ser o ideal se app.py estiver em outro lugar.
    # Melhor usar caminhos relativos ao script ou uma configuração mais robusta.
    # Para simplificar, vamos assumir que os arquivos são salvos na raiz do projeto por enquanto.
    # No entanto, para melhor organização, um subdiretório seria ideal.
    # path_base = os.path.join(os.path.dirname(__file__), "..", "assistentes_salvos") # Exemplo se utils.py está em frontend/
    path_base = ASSISTENTES_SAVE_DIR # Usar o diretório definido
    os.makedirs(path_base, exist_ok=True) # Garante que o diretório exista

    safe_nome_assistente = nome_assistente.replace(' ', '_').lower().strip()
    instrucoes_file = os.path.join(path_base, f"assistente_{safe_nome_assistente}_instrucoes.txt")
    faiss_file = os.path.join(path_base, f"assistente_{safe_nome_assistente}_faiss.index")
    chunks_file = os.path.join(path_base, f"assistente_{safe_nome_assistente}_chunks.json")
    # Arquivo para armazenar nomes dos arquivos originais associados aos chunks/FAISS
    uploaded_files_info_file = os.path.join(path_base, f"assistente_{safe_nome_assistente}_uploaded_files.json") 

    loaded_something = False
    if os.path.exists(instrucoes_file):
        try:
            with open(instrucoes_file, "r", encoding="utf-8") as f:
                st.session_state["instrucoes_finais"] = f.read()
            loaded_something = True
        except Exception as e:
            st.error(f"Erro ao carregar instruções para '{nome_assistente}': {e}")
    
    if os.path.exists(chunks_file) and os.path.exists(faiss_file):
        try:
            with open(chunks_file, "r", encoding="utf-8") as f:
                st.session_state["doc_chunks"] = json.load(f)
            st.session_state["faiss_index"] = faiss.read_index(faiss_file)
            if os.path.exists(uploaded_files_info_file):
                with open(uploaded_files_info_file, "r", encoding="utf-8") as f_info:
                    st.session_state["uploaded_files"] = json.load(f_info) # Carrega lista de nomes
            loaded_something = True
        except Exception as e:
            st.error(f"Erro ao carregar chunks, índice FAISS ou info de arquivos para '{nome_assistente}': {e}")
            st.session_state["doc_chunks"] = []
            st.session_state["faiss_index"] = inicializar_faiss()
            st.session_state["uploaded_files"] = []
    elif os.path.exists(chunks_file) and not os.path.exists(faiss_file):
        st.warning(f"Chunks para '{nome_assistente}' encontrados, mas índice FAISS não. Documentos podem precisar ser reprocessados ou o índice recriado.")
        try:
            with open(chunks_file, "r", encoding="utf-8") as f:
                st.session_state["doc_chunks"] = json.load(f)
            # Tentar recriar o índice FAISS se houver chunks e API key
            if st.session_state["doc_chunks"] and openai_api_key:
                st.info(f"Tentando recriar índice FAISS para '{nome_assistente}' a partir dos chunks existentes...")
                embeddings = gerar_embeddings(st.session_state["doc_chunks"], openai_api_key)
                if embeddings:
                    new_index = inicializar_faiss()
                    for emb in embeddings:
                        new_index.add(np.expand_dims(emb, axis=0))
                    st.session_state["faiss_index"] = new_index
                    faiss.write_index(new_index, faiss_file) # Salva o índice recriado
                    st.success(f"Índice FAISS para '{nome_assistente}' recriado e salvo.")
                else:
                    st.error(f"Não foi possível recriar o índice FAISS para '{nome_assistente}'.")
        except Exception as e:
            st.error(f"Erro ao carregar chunks ou tentar recriar FAISS para '{nome_assistente}': {e}")

    st.session_state["mem0_instance"] = inicializar_mem0(username, nome_assistente, mem0_api_key)
    
    if st.session_state["mem0_instance"]:
        try:
            # Recupera o histórico de chat do mem0
            mem0_history = st.session_state["mem0_instance"].get_history()
            # Converte o histórico do mem0 para o formato esperado pelo st.session_state["chat_history"]
            # Assumindo que o histórico do mem0 retorna uma lista de objetos com 'role' e 'content'
            st.session_state["chat_history"] = [{"role": msg.role, "content": msg.content} for msg in mem0_history]
            if mem0_history:
                st.info(f"Histórico de chat carregado do Mem0 para '{nome_assistente}'.")
        except Exception as e:
            st.error(f"Erro ao carregar histórico do Mem0 para '{nome_assistente}': {e}")

    if loaded_something:
        st.success(f"Dados para o assistente '{nome_assistente}' carregados.")
    elif nome_assistente and nome_assistente not in ["Nenhum Assistente Salvo", "Nenhum"]:
        # Se nenhum arquivo foi carregado, mas um nome de assistente foi fornecido (e não é um placeholder)
        st.info(f"Nenhum dado salvo encontrado para '{nome_assistente}'. Iniciando com configuração padrão.")

    # Define o diretório 'assistentes_salvos' dentro do diretório 'frontend'
    # Assume que utils.py está em frontend/, então os.getcwd() pode não ser o ideal se app.py estiver em outro lugar.
    # Melhor usar caminhos relativos ao script ou uma configuração mais robusta.
    # Para simplificar, vamos assumir que os arquivos são salvos na raiz do projeto por enquanto.
    # No entanto, para melhor organização, um subdiretório seria ideal.
    # path_base = os.path.join(os.path.dirname(__file__), "..", "assistentes_salvos") # Exemplo se utils.py está em frontend/
    path_base = ASSISTENTES_SAVE_DIR # Usar o diretório definido
    os.makedirs(path_base, exist_ok=True) # Garante que o diretório exista

    safe_nome_assistente = nome_assistente.replace(' ', '_').lower().strip()
    instrucoes_file = os.path.join(path_base, f"assistente_{safe_nome_assistente}_instrucoes.txt")
    faiss_file = os.path.join(path_base, f"assistente_{safe_nome_assistente}_faiss.index")
    chunks_file = os.path.join(path_base, f"assistente_{safe_nome_assistente}_chunks.json")
    # Arquivo para armazenar nomes dos arquivos originais associados aos chunks/FAISS
    uploaded_files_info_file = os.path.join(path_base, f"assistente_{safe_nome_assistente}_uploaded_files.json") 

    loaded_something = False
    if os.path.exists(instrucoes_file):
        try:
            with open(instrucoes_file, "r", encoding="utf-8") as f:
                st.session_state["instrucoes_finais"] = f.read()
            loaded_something = True
        except Exception as e:
            st.error(f"Erro ao carregar instruções para '{nome_assistente}': {e}")
    
    if os.path.exists(chunks_file) and os.path.exists(faiss_file):
        try:
            with open(chunks_file, "r", encoding="utf-8") as f:
                st.session_state["doc_chunks"] = json.load(f)
            st.session_state["faiss_index"] = faiss.read_index(faiss_file)
            if os.path.exists(uploaded_files_info_file):
                with open(uploaded_files_info_file, "r", encoding="utf-8") as f_info:
                    st.session_state["uploaded_files"] = json.load(f_info) # Carrega lista de nomes
            loaded_something = True
        except Exception as e:
            st.error(f"Erro ao carregar chunks, índice FAISS ou info de arquivos para '{nome_assistente}': {e}")
            st.session_state["doc_chunks"] = []
            st.session_state["faiss_index"] = inicializar_faiss()
            st.session_state["uploaded_files"] = []
    elif os.path.exists(chunks_file) and not os.path.exists(faiss_file):
        st.warning(f"Chunks para '{nome_assistente}' encontrados, mas índice FAISS não. Documentos podem precisar ser reprocessados ou o índice recriado.")
        try:
            with open(chunks_file, "r", encoding="utf-8") as f:
                st.session_state["doc_chunks"] = json.load(f)
            # Tentar recriar o índice FAISS se houver chunks e API key
            if st.session_state["doc_chunks"] and openai_api_key:
                st.info(f"Tentando recriar índice FAISS para '{nome_assistente}' a partir dos chunks existentes...")
                embeddings = gerar_embeddings(st.session_state["doc_chunks"], openai_api_key)
                if embeddings:
                    new_index = inicializar_faiss()
                    for emb in embeddings:
                        new_index.add(np.expand_dims(emb, axis=0))
                    st.session_state["faiss_index"] = new_index
                    faiss.write_index(new_index, faiss_file) # Salva o índice recriado
                    st.success(f"Índice FAISS para '{nome_assistente}' recriado e salvo.")
                else:
                    st.error(f"Não foi possível recriar o índice FAISS para '{nome_assistente}'.")
        except Exception as e:
            st.error(f"Erro ao carregar chunks ou tentar recriar FAISS para '{nome_assistente}': {e}")

    st.session_state["mem0_instance"] = inicializar_mem0(username, nome_assistente, mem0_api_key)
    
    if loaded_something:
        st.success(f"Dados para o assistente '{nome_assistente}' carregados.")
    elif nome_assistente and nome_assistente not in ["Nenhum Assistente Salvo", "Nenhum"]:
        # Se nenhum arquivo foi carregado, mas um nome de assistente foi fornecido (e não é um placeholder)
        st.info(f"Nenhum dado salvo encontrado para '{nome_assistente}'. Iniciando com configuração padrão.")

# Define o diretório base para salvar/carregar dados dos assistentes
# __file__ é o caminho para utils.py. Queremos ..\frontend\assistentes_salvos a partir daqui.
# No entanto, como utils.py está em frontend, o diretório de assistentes deve ser um subdiretório de frontend.
ASSISTENTES_SAVE_DIR = os.path.join(os.path.dirname(__file__), "assistentes_salvos")
os.makedirs(ASSISTENTES_SAVE_DIR, exist_ok=True)

def get_assistentes_existentes() -> List[str]:
    """Lista assistentes existentes baseados nos arquivos de instruções salvos."""
    path_base = ASSISTENTES_SAVE_DIR
    assistentes = []
    try:
        # Procura por arquivos de instrução no diretório atual
        for f_path in glob.glob(os.path.join(path_base, "assistente_*_instrucoes.txt")):
            filename = os.path.basename(f_path)
            if filename.startswith("assistente_") and filename.endswith("_instrucoes.txt"):
                # Extrai: assistente_NOME_DO_ASSISTENTE_instrucoes.txt -> NOME_DO_ASSISTENTE
                nome_assistente_safe = filename[len("assistente_"):-len("_instrucoes.txt")]
                # Reverter a transformação de 'safe_nome_assistente' para o nome original pode ser complexo
                # se o nome original continha espaços ou maiúsculas/minúsculas mistas.
                # Por simplicidade, vamos assumir que o nome salvo (safe) é o que será listado.
                # Para uma melhor UX, seria bom salvar o nome original em algum lugar (ex: no arquivo de config do assistente)
                # e ler de lá. Por agora, usamos o nome do arquivo.
                assistentes.append(nome_assistente_safe.replace('_', ' ').title()) # Tentativa de reverter para um formato mais legível
    except Exception as e:
        st.error(f"Erro ao listar assistentes existentes: {e}")
    
    if not assistentes:
        return ["Nenhum Assistente Salvo"]
    return sorted(list(set(assistentes))) # Remove duplicatas e ordena

def reset_session():
    """Reseta chaves específicas do estado da sessão."""
    keys_to_reset = [
        "username", "assistente_selecionado", "assistente_novo", 
        "chat_history", "chat_mode", "uploaded_files", "assistente_config", 
        "faiss_index", "doc_chunks", "mem0_instance", "instrucoes_finais",
        "config_flow_complete", "current_config_step_key", 'config_flow_initial_message_shown',
        "loading_ia"
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]
    # st.info("Sessão resetada.") # Opcional: feedback ao usuário