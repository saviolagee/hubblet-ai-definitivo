import streamlit as st
import requests
from typing import List, Dict
from datetime import datetime
import time
import os
import json # Adicionado para salvar/carregar metadados de arquivos
from dotenv import load_dotenv
import numpy as np
from mem0 import MemoryClient
from openai import OpenAI # Para uso direto na IA de configuração

# Importações de utils (ajustado para importação direta)
from utils import (
    get_assistentes_existentes,
    reset_session,
    inicializar_faiss,
    gerar_embeddings,
    processar_arquivos,
    carregar_ou_inicializar_dados_assistente,
    load_chat_history,  # Adicionado
    save_chat_history,  # Adicionado
    create_new_chat_session,  # Adicionado
    list_chat_sessions,  # Adicionado
    get_chat_session_messages,  # Adicionado
    add_message_to_session  # Adicionado
)

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

st.set_page_config(page_title="Hubblet: Criação de Assistente", layout="wide")

st.markdown("<div style='height:3.5rem;'></div>", unsafe_allow_html=True)

# Constantes para o fluxo de configuração do assistente
CONFIG_STEPS_ORDER = ["nome", "estilo", "funcoes", "fontes_info"]
CONFIG_QUESTIONS = {
    "nome": "Qual será o nome do seu assistente?",
    "estilo": "Qual estilo de comunicação você prefere para o assistente? (Ex: formal, amigável, técnico, divertido)",
    "funcoes": "Quais são as principais funções ou tarefas que este assistente deve realizar?",
    "fontes_info": "Você já anexou os documentos que o assistente usará como base de conhecimento na coluna ao lado? Responda 'sim' ou 'não'. Se não precisar adicionar agora, pode dizer 'não'."
}

# Página de Login/Seleção de Assistente
def pagina_login():
    st.title("Hubblet AI - Login e Seleção de Assistente")
    st.write("Entre com seu usuário e escolha um assistente ou crie um novo.")
    with st.form(key="login_form"):
        username = st.text_input("Usuário", value=st.session_state.get("username", ""))
        assistentes = get_assistentes_existentes() + ["Criar novo assistente"]
        assistente_selecionado_login_val = st.selectbox("Escolha um assistente", assistentes, index=0, key="assistente_login_selectbox")
        submit = st.form_submit_button("Entrar")
    if submit:
        if not username.strip():
            st.warning("Usuário é obrigatório.")
            return
        st.session_state["username"] = username.strip()
        st.session_state["assistente_selecionado_login"] = assistente_selecionado_login_val # Salva o assistente escolhido no login
        openai_api_key = os.environ.get("OPENAI_API_KEY", "")

        if assistente_selecionado_login_val == "Criar novo assistente":
            st.session_state["menu_sidebar"] = "Configuração/Chat"
            st.session_state["assistente_selecionado"] = None # Indica novo assistente
            carregar_ou_inicializar_dados_assistente(username=st.session_state["username"], nome_assistente="", openai_api_key=openai_api_key)
            st.session_state["chat_mode"] = "criar"
            st.session_state["assistente_config"] = {}
            st.session_state["instrucoes_finais"] = None
            st.session_state['config_flow_initial_message_shown'] = False
            st.session_state['config_flow_complete'] = False
            st.session_state['current_config_step_key'] = CONFIG_STEPS_ORDER[0]
            st.session_state["config_chat_history"] = [] 

        else: # Assistente existente selecionado
            st.session_state["assistente_selecionado"] = assistente_selecionado_login_val
            st.session_state["menu_sidebar"] = "Chat Principal"
            st.session_state["chat_mode"] = "editar"
            carregar_ou_inicializar_dados_assistente(username=st.session_state["username"], nome_assistente=assistente_selecionado_login_val, openai_api_key=openai_api_key)
            
            sessions = list_chat_sessions(user_id=st.session_state["username"])
            # Filtra sessões para mostrar apenas aquelas relacionadas ao assistente logado
            sessoes_do_assistente_logado = [s for s in sessions if s.get("title", "").startswith(f"Chat com {assistente_selecionado_login_val}")]

            if sessoes_do_assistente_logado:
                # Carrega a sessão mais recente do assistente logado
                latest_session = sorted(sessoes_do_assistente_logado, key=lambda s: s.get("updated_at", ""), reverse=True)[0]
                st.session_state["current_chat_session_id"] = latest_session["id"]
                st.session_state["chat_principal_history"] = get_chat_session_messages(latest_session["id"])
            else:
                # Cria uma nova sessão se não houver nenhuma para este assistente
                new_session_title = f"Chat com {assistente_selecionado_login_val}"
                new_session = create_new_chat_session(user_id=st.session_state["username"], title=new_session_title)
                st.session_state["current_chat_session_id"] = new_session["id"]
                st.session_state["chat_principal_history"] = []
        
        st.rerun()

# Página de Criação/Atualização do Assistente
def pagina_chat_assistente():
    with st.sidebar:
        st.title(f"Hubblet AI")
        st.write(f"Usuário: **{st.session_state.get('username','')}**")
        st.divider()
        st.subheader("Navegação")
        if st.button("Ir para o Chat Principal", key="goto_chat_btn"):
            st.session_state["menu_sidebar"] = "Chat Principal"
            st.rerun()
        st.divider()

    col_chat, col_upload = st.columns([1.3, 1])
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")

    if "assistente_config" not in st.session_state: st.session_state["assistente_config"] = {}
    if "config_chat_history" not in st.session_state: st.session_state["config_chat_history"] = []
    if "instrucoes_finais" not in st.session_state: st.session_state["instrucoes_finais"] = None

    is_editing = st.session_state.get("chat_mode") == "editar" and st.session_state.get("assistente_selecionado")

    if "config_flow_complete" not in st.session_state:
        st.session_state["config_flow_complete"] = True if is_editing and st.session_state.get("instrucoes_finais") else False
    
    if "current_config_step_key" not in st.session_state:
        st.session_state["current_config_step_key"] = None if st.session_state["config_flow_complete"] else CONFIG_STEPS_ORDER[0]

    if 'config_flow_initial_message_shown' not in st.session_state:
        st.session_state['config_flow_initial_message_shown'] = False

    if not st.session_state['config_flow_initial_message_shown']:
        if is_editing and st.session_state.get("instrucoes_finais"):
            st.session_state["config_flow_complete"] = True 
            st.session_state["current_config_step_key"] = None
            edit_message = f"Editando '{st.session_state['assistente_selecionado']}'. As instruções atuais estão carregadas. Você pode refiná-las ou pedir para gerar uma nova versão."
            if not st.session_state["config_chat_history"] or st.session_state["config_chat_history"][-1]["content"] != edit_message:
                st.session_state["config_chat_history"].append({"role": "assistant", "content": edit_message})
        elif not is_editing:
            st.session_state['current_config_step_key'] = CONFIG_STEPS_ORDER[0]
            welcome_message = f"Olá! Eu sou o Assistente de Configuração da Hubblet e estou aqui para te ajudar a criar seu novo assistente personalizado. Vamos começar com algumas perguntas para definir a base dele. A primeira é: {CONFIG_QUESTIONS[CONFIG_STEPS_ORDER[0]]}"
            st.session_state["config_chat_history"].append({"role": "assistant", "content": welcome_message})
        st.session_state['config_flow_initial_message_shown'] = True
    
    if not is_editing and not st.session_state["config_flow_complete"]:
        next_step_key = None
        for step_key in CONFIG_STEPS_ORDER:
            if step_key not in st.session_state["assistente_config"]:
                next_step_key = step_key
                break
        
        if next_step_key is None:
            st.session_state["config_flow_complete"] = True
            st.session_state["current_config_step_key"] = None
            completion_message = "Configuração básica completa! Agora você pode refinar os detalhes com a IA ou pedir para gerar as instruções finais. O que gostaria de fazer?"
            if not st.session_state["config_chat_history"] or st.session_state["config_chat_history"][-1]["content"] != completion_message:
                st.session_state["config_chat_history"].append({"role": "assistant", "content": completion_message})
        else:
            st.session_state["current_config_step_key"] = next_step_key
            question_to_ask = CONFIG_QUESTIONS[st.session_state["current_config_step_key"]]
            if not st.session_state["config_chat_history"] or st.session_state["config_chat_history"][-1]["content"] != question_to_ask or st.session_state["config_chat_history"][-1]["role"] == "user":
                st.session_state["config_chat_history"].append({"role": "assistant", "content": question_to_ask})

    with col_chat:
        st.markdown("<div style='font-size:1.3rem;font-weight:600;margin-bottom:0.5rem;'>Chat de Configuração</div>", unsafe_allow_html=True)
        if not openai_api_key:
            st.warning("⚠️ Defina a variável de ambiente OPENAI_API_KEY para habilitar o chat com a IA.")
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state["config_chat_history"]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        with st.form(key="chat_form_config", clear_on_submit=True):
            prompt_placeholder = "Sua resposta..." if not st.session_state["config_flow_complete"] else "Refine os detalhes ou peça para gerar as instruções..."
            prompt = st.text_input("", placeholder=prompt_placeholder, key="config_prompt_input")
            enviar = st.form_submit_button("Enviar", help="Enviar mensagem", use_container_width=True)

        if enviar and prompt.strip():
            st.session_state["config_chat_history"].append({"role": "user", "content": prompt.strip()})

            if not st.session_state["config_flow_complete"] and st.session_state["current_config_step_key"]:
                current_key = st.session_state["current_config_step_key"]
                st.session_state["assistente_config"][current_key] = prompt.strip()
                if current_key == "nome": 
                    st.session_state["assistente_config"]["nome"] = prompt.strip()
            else:
                if not openai_api_key:
                    st.session_state["config_chat_history"].append({"role": "assistant", "content": "OPENAI_API_KEY não configurada. Não posso processar este pedido."})
                else:
                    try:
                        client = OpenAI(api_key=openai_api_key)
                        # Prepara o contexto para a IA de configuração
                        config_context = []
                        # Adiciona uma instrução de sistema para a IA de configuração
                        system_prompt_config = "Você é um assistente de IA ajudando um usuário a configurar um novo assistente de IA. " \
                                             "O usuário fornecerá informações sobre o nome, estilo, funções e fontes de informação do assistente. " \
                                             "Seu objetivo é ajudar o usuário a refinar esses detalhes e, eventualmente, gerar um conjunto conciso de instruções finais para o novo assistente. " \
                                             "Se o usuário pedir para 'gerar instruções', use as informações coletadas para criar um prompt de sistema para o novo assistente."
                        config_context.append({"role": "system", "content": system_prompt_config})
                        
                        # Adiciona o histórico do chat de configuração
                        config_context.extend(st.session_state["config_chat_history"])
                        
                        # Adiciona informações já coletadas ao contexto, se houver
                        if st.session_state["assistente_config"]:
                            collected_info = "\n\nInformações coletadas até agora:\n"
                            for key, value in st.session_state["assistente_config"].items():
                                collected_info += f"- {key.capitalize()}: {value}\n"
                            # Adiciona como uma mensagem de sistema ou anexa à existente
                            config_context.append({"role": "system", "content": collected_info})

                        if st.session_state.get("instrucoes_finais"):
                             config_context.append({"role": "system", "content": f"Instruções atuais (se estiver editando): {st.session_state['instrucoes_finais']}"}) 

                        with st.spinner("Processando..."):
                            response = client.chat.completions.create(
                                model="gpt-3.5-turbo", 
                                messages=config_context,
                                temperature=0.5
                            )
                            assistant_response_config = response.choices[0].message.content
                            st.session_state["config_chat_history"].append({"role": "assistant", "content": assistant_response_config})
                            
                            # Verifica se a IA gerou as instruções finais
                            if "instruções finais geradas" in assistant_response_config.lower() or "prompt de sistema gerado" in assistant_response_config.lower() or (prompt.lower().startswith("gerar instruções") and len(assistant_response_config) > 50):
                                # Heurística para detectar se a IA gerou as instruções
                                # Idealmente, a IA sinalizaria isso de forma mais explícita
                                st.session_state["instrucoes_finais"] = assistant_response_config # Assume que a resposta é o prompt
                                st.success("Instruções finais geradas/atualizadas!")

                    except Exception as e:
                        st.error(f"Erro ao comunicar com a IA de configuração: {e}")
                        st.session_state["config_chat_history"].append({"role": "assistant", "content": f"Erro: {e}"})
            st.rerun()

    with col_upload:
        st.markdown("<div style='font-size:1.3rem;font-weight:600;margin-bottom:0.5rem;'>Base de Conhecimento (Opcional)</div>", unsafe_allow_html=True)
        uploaded_file_objects = st.file_uploader(
            "Adicione arquivos (.txt, .md, .pdf) para o assistente usar como referência.", 
            accept_multiple_files=True, 
            type=['txt', 'md', 'pdf'],
            key="file_uploader_config"
        )

        if uploaded_file_objects:
            # Processa apenas os arquivos que ainda não foram processados
            novos_arquivos_para_processar = []
            nomes_ja_processados = st.session_state.get("uploaded_files", [])
            for ufo in uploaded_file_objects:
                if ufo.name not in nomes_ja_processados:
                    novos_arquivos_para_processar.append(ufo)
            
            if novos_arquivos_para_processar:
                if not openai_api_key:
                    st.warning("OPENAI_API_KEY não definida. Não é possível processar arquivos.")
                else:
                    chunks, embeddings, nomes_processados = processar_arquivos(novos_arquivos_para_processar, openai_api_key)
                    if chunks and embeddings:
                        st.session_state["doc_chunks"].extend(chunks)
                        for emb in embeddings:
                            st.session_state["faiss_index"].add(np.expand_dims(emb, axis=0))
                        st.session_state["uploaded_files"].extend(nomes_processados)
                        st.success(f"{len(nomes_processados)} novo(s) arquivo(s) processado(s) e adicionado(s) à base de conhecimento.")
                    elif nomes_processados: # Arquivos foram lidos mas não geraram embeddings
                        st.warning(f"{len(nomes_processados)} arquivo(s) lido(s), mas falha ao gerar embeddings. Verifique o conteúdo e a chave da API.")

        st.subheader("Arquivos Carregados:")
        if st.session_state.get("uploaded_files"):
            for nome_arq in st.session_state["uploaded_files"]:
                st.info(f"- {nome_arq}")
        else:
            st.info("Nenhum arquivo carregado ainda.")
        
        st.divider()
        st.subheader("Instruções Finais do Assistente:")
        if st.session_state.get("instrucoes_finais"):
            st.text_area("", value=st.session_state["instrucoes_finais"], height=200, disabled=True, key="instrucoes_display")
        else:
            st.info("As instruções finais serão geradas ou refinadas através do chat de configuração.")

        if st.button("Salvar Assistente", key="save_assistant_btn"):
            nome_assistente_config = st.session_state.get("assistente_config", {}).get("nome")
            if not nome_assistente_config or not nome_assistente_config.strip():
                st.error("O nome do assistente é obrigatório para salvar.")
            elif not st.session_state.get("instrucoes_finais"):
                # Tenta obter as instruções da última mensagem do assistente no chat de configuração
                # se não foram explicitamente marcadas como 'instrucoes_finais'
                last_assistant_message = None
                if st.session_state.get("config_chat_history"):
                    for msg in reversed(st.session_state.get("config_chat_history", [])):
                        if msg["role"] == "assistant":
                            last_assistant_message = msg["content"]
                            break
                
                if last_assistant_message and len(last_assistant_message) > 50: # Heurística: mensagem longa do assistente
                    st.session_state["instrucoes_finais"] = last_assistant_message
                    st.info("Instruções finais inferidas da última resposta do chat de configuração.")
                else:
                    st.error("As instruções finais do assistente são obrigatórias para salvar. Use o chat para gerá-las ou certifique-se que a IA as confirmou.")
                    return # Impede o salvamento se ainda não houver instruções
            
            # Prossegue para o salvamento se as instruções foram obtidas (diretamente ou inferidas)
            if st.session_state.get("instrucoes_finais"):
                try:
                    # Salvar instruções
                    # Garante que o caminho seja relativo ao diretório do frontend, onde utils.py espera.
                    path_base = os.path.join(os.path.dirname(__file__), "assistentes_salvos")
                    os.makedirs(path_base, exist_ok=True)
                    safe_nome_assistente = nome_assistente_config.replace(' ', '_').lower().strip()
                    
                    config_file_md = os.path.join(path_base, f"assistente_{safe_nome_assistente}_config.md")
                    with open(config_file_md, "w", encoding="utf-8") as f:
                        # Adiciona um cabeçalho simples ou apenas salva as instruções diretamente
                        # Se for salvar apenas as instruções, pode ser f.write(st.session_state["instrucoes_finais"])
                        # Para manter um formato que possa ser estendido no futuro, podemos adicionar um marcador.
                        f.write(f"# Configuração do Assistente: {nome_assistente_config}\n\n")
                        f.write("## Instruções Finais:\n")
                        f.write(st.session_state["instrucoes_finais"])

                    # Adicionar instruções finais ao FAISS index
                    if st.session_state.get("instrucoes_finais") and openai_api_key:
                        instrucoes_texto = st.session_state["instrucoes_finais"]
                        # Usar a função gerar_embeddings de utils.py (precisa ser importada ou chamada via processar_arquivos)
                        # Para simplificar aqui, vamos chamar a lógica de embedding diretamente se possível
                        # ou garantir que processar_arquivos pode lidar com texto direto.
                         # Assumindo que gerar_embeddings está acessível ou podemos replicar a chamada:
                         
                        instrucoes_embeddings = gerar_embeddings([instrucoes_texto], openai_api_key)
                        if instrucoes_embeddings:
                            if "doc_chunks" not in st.session_state:
                                st.session_state["doc_chunks"] = []
                            if "faiss_index" not in st.session_state or st.session_state["faiss_index"] is None:
                                # A dimensão do embedding é 1536 para text-embedding-ada-002
                                st.session_state["faiss_index"] = faiss.IndexFlatL2(1536) 

                            st.session_state["doc_chunks"].append(f"Instruções do Assistente: {instrucoes_texto}") # Adiciona com um prefixo
                            st.session_state["faiss_index"].add(np.array(instrucoes_embeddings[0], dtype=np.float32).reshape(1, -1))
                            st.info("Instruções do assistente adicionadas ao índice de conhecimento.")
                        else:
                            st.warning("Não foi possível gerar embeddings para as instruções do assistente.")
                    
                    # Salvar FAISS e chunks se existirem (agora pode incluir as instruções)
                    if st.session_state.get("faiss_index") and st.session_state["faiss_index"].ntotal > 0:
                        faiss_file = os.path.join(path_base, f"assistente_{safe_nome_assistente}_faiss.index")
                        faiss.write_index(st.session_state["faiss_index"], faiss_file)
                        
                        chunks_file = os.path.join(path_base, f"assistente_{safe_nome_assistente}_chunks.json")
                        with open(chunks_file, "w", encoding="utf-8") as f:
                            json.dump(st.session_state.get("doc_chunks", []), f)
                        
                        # Salvar nomes dos arquivos originais
                        uploaded_files_info_file = os.path.join(path_base, f"assistente_{safe_nome_assistente}_uploaded_files.json")
                        with open(uploaded_files_info_file, "w", encoding="utf-8") as f_info:
                            json.dump(st.session_state.get("uploaded_files", []), f_info)

                    st.success(f"Assistente '{nome_assistente_config}' salvo com sucesso!")
                    st.session_state["assistente_selecionado"] = nome_assistente_config # Define como selecionado
                    st.session_state["menu_sidebar"] = "Chat Principal" # Muda para o chat principal
                    # Garante que a sessão de chat para este assistente seja criada/carregada
                    sessions = list_chat_sessions(user_id=st.session_state["username"])
                    session_title_chat = f"Chat com {nome_assistente_config}"
                    existing_chat_session = next((s for s in sessions if s.get("title") == session_title_chat), None)
                    if existing_chat_session:
                        st.session_state["current_chat_session_id"] = existing_chat_session["id"]
                    else:
                        new_chat_session = create_new_chat_session(user_id=st.session_state["username"], title=session_title_chat)
                        st.session_state["current_chat_session_id"] = new_chat_session["id"]
                    st.session_state["chat_principal_history"] = get_chat_session_messages(st.session_state["current_chat_session_id"])
                    st.rerun()

                except Exception as e:
                    st.error(f"Erro ao salvar o assistente: {e}")

# Página de Chat Principal
def pagina_chat_principal():
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    # Adicionar inicialização do Mem0 Client
    mem0_client = MemoryClient() # Assumindo que a configuração (ex: API key) é feita via variáveis de ambiente se necessário

    if "username" not in st.session_state or not st.session_state["username"]:
        st.warning("Por favor, faça login primeiro.")
        st.session_state["menu_sidebar"] = "Login"
        st.rerun()
        return

    # Inicializa o estado da sessão para o chat principal, se necessário
    if "chat_principal_history" not in st.session_state: st.session_state["chat_principal_history"] = []
    if "current_chat_session_id" not in st.session_state:
        sessions = list_chat_sessions(user_id=st.session_state["username"])
        assistente_logado = st.session_state.get("assistente_selecionado_login", "Chat Geral") # Usa o assistente do login
        session_title = f"Chat com {assistente_logado}"
        
        # Procura por sessões existentes com este assistente
        sessoes_do_assistente = [s for s in sessions if s.get("title", "").startswith(session_title)]
        
        if sessoes_do_assistente:
            latest_session_for_assistant = sorted(sessoes_do_assistente, key=lambda s: s.get("updated_at", ""), reverse=True)[0]
            st.session_state["current_chat_session_id"] = latest_session_for_assistant["id"]
            st.session_state["chat_principal_history"] = get_chat_session_messages(latest_session_for_assistant["id"])
        else: 
            # Cria uma nova sessão para este assistente se nenhuma existir
            new_session = create_new_chat_session(user_id=st.session_state["username"], title=session_title)
            st.session_state["current_chat_session_id"] = new_session["id"]
            st.session_state["chat_principal_history"] = []

    with st.sidebar:
        st.title(f"Hubblet AI")
        st.write(f"Usuário: **{st.session_state.get('username','')}**")
        assistente_ativo = st.session_state.get('assistente_selecionado', 'N/A')
        st.write(f"Assistente: **{assistente_ativo}**")
        st.divider()
        st.subheader("Navegação")
        if st.button("Configurar/Editar Assistente", key="goto_config_btn"):
            st.session_state["menu_sidebar"] = "Configuração/Chat"
            if st.session_state.get("assistente_selecionado") and st.session_state.get("assistente_selecionado") != "Criar novo assistente":
                 carregar_ou_inicializar_dados_assistente(username=st.session_state["username"], nome_assistente=st.session_state["assistente_selecionado"], openai_api_key=openai_api_key)
                 st.session_state["chat_mode"] = "editar"
            else: 
                 carregar_ou_inicializar_dados_assistente(username=st.session_state["username"], nome_assistente="", openai_api_key=openai_api_key)
                 st.session_state["chat_mode"] = "criar"
                 st.session_state["assistente_config"] = {}
                 st.session_state["instrucoes_finais"] = None
                 st.session_state['config_flow_initial_message_shown'] = False
                 st.session_state['config_flow_complete'] = False
                 st.session_state['current_config_step_key'] = CONFIG_STEPS_ORDER[0]
                 st.session_state["config_chat_history"] = [] 
            st.rerun()

        if st.button("Logout", key="logout_btn_chat"):
            reset_session()
            st.rerun()
        st.divider()
        st.subheader("Minhas Conversas")

        if st.button(":heavy_plus_sign: Nova Conversa", key="new_chat_btn_sidebar", use_container_width=True, type="primary"):
            assistente_logado = st.session_state.get("assistente_selecionado_login", None)
            if not assistente_logado or assistente_logado == "Criar novo assistente":
                st.warning("Por favor, selecione um assistente no login para iniciar uma nova conversa.")
            else:
                default_title = f"Chat com {assistente_logado}"
                user_sessions_check = list_chat_sessions(user_id=st.session_state["username"])
                existing_titles = [s.get("title") for s in user_sessions_check]
                count = 1
                new_title = default_title
                if new_title in existing_titles:
                    new_title_candidate = f"{default_title} ({count})"
                    while new_title_candidate in existing_titles:
                        count += 1
                        new_title_candidate = f"{default_title} ({count})"
                    new_title = new_title_candidate

                new_session_obj = create_new_chat_session(user_id=st.session_state["username"], title=new_title)
                st.session_state["current_chat_session_id"] = new_session_obj["id"]
                st.session_state["chat_principal_history"] = []
                if st.session_state.get("assistente_selecionado") != assistente_logado:
                    st.session_state["assistente_selecionado"] = assistente_logado
                    carregar_ou_inicializar_dados_assistente(username=st.session_state["username"],
                                                           nome_assistente=assistente_logado,
                                                           openai_api_key=openai_api_key)
                st.rerun()

        assistente_logado_sidebar = st.session_state.get("assistente_selecionado_login", None)
        user_sessions = list_chat_sessions(user_id=st.session_state["username"])

        sessoes_visiveis = []
        if assistente_logado_sidebar and assistente_logado_sidebar != "Criar novo assistente":
            sessoes_visiveis = [s for s in user_sessions if s.get("title", "").startswith(f"Chat com {assistente_logado_sidebar}")]
        
        if sessoes_visiveis:
            for session in sorted(sessoes_visiveis, key=lambda s: s.get("updated_at", ""), reverse=True):
                session_display_name = session.get("title", session["id"][:8])
                is_active_session = session["id"] == st.session_state.get("current_chat_session_id")
                button_type = "primary" if is_active_session else "secondary"
                
                if st.button(f":chat_bubble_outline: {session_display_name}", key=f"session_btn_{session['id']}", help=f"Abrir '{session.get('title', 'Conversa')}'", type=button_type, use_container_width=True):
                    st.session_state["current_chat_session_id"] = session["id"]
                    st.session_state["chat_principal_history"] = get_chat_session_messages(session["id"])
                    if st.session_state.get("assistente_selecionado") != assistente_logado_sidebar:
                        st.session_state["assistente_selecionado"] = assistente_logado_sidebar
                        carregar_ou_inicializar_dados_assistente(username=st.session_state["username"], 
                                                               nome_assistente=assistente_logado_sidebar, 
                                                               openai_api_key=openai_api_key)
                    st.rerun()
        elif assistente_logado_sidebar and assistente_logado_sidebar != "Criar novo assistente":
            st.info(f"Nenhuma conversa com '{assistente_logado_sidebar}' ainda.")
        else:
            st.info("Selecione um assistente no login para ver as conversas.")


    st.markdown(f"<div style='font-size:1.3rem;font-weight:600;margin-bottom:0.5rem;'>Chat com {st.session_state.get('assistente_selecionado', 'Assistente')}</div>", unsafe_allow_html=True)

    chat_container_principal = st.container()
    with chat_container_principal:
        for msg in st.session_state["chat_principal_history"]:
            role = msg.get("role")
            content = msg.get("content")
            if role and content:
                with st.chat_message(role):
                    st.markdown(content)

    prompt_principal = st.chat_input(f"Pergunte ao {st.session_state.get('assistente_selecionado', 'Assistente')}...")

    if prompt_principal:
        if not openai_api_key:
            st.warning("OPENAI_API_KEY não definida. Não é possível gerar resposta.")
            st.stop()

        add_message_to_session(st.session_state["current_chat_session_id"], "user", prompt_principal)
        st.session_state["chat_principal_history"].append({"role": "user", "content": prompt_principal})
        
        with st.chat_message("user"):
            st.markdown(prompt_principal)

        contexto_chat_ia = []
        if st.session_state.get("instrucoes_finais"):
            contexto_chat_ia.append({"role": "system", "content": st.session_state["instrucoes_finais"]})
        
        current_user_id = st.session_state["username"]
        current_agent_id = st.session_state.get('assistente_selecionado')
        
        # Adicionar busca de memória do mem0 AQUI
        if mem0_client and current_user_id:
            try:
                retrieved_memories = mem0_client.search(
                    query=prompt_principal,
                    user_id=current_user_id
                )
                # st.write(f"Memórias recuperadas mem0: {retrieved_memories}") # Para debug
                if retrieved_memories and isinstance(retrieved_memories, list):
                    # Supondo que retrieved_memories é uma lista de dicts e cada dict tem uma chave 'memory'
                    memories_text_list = [mem.get("memory") for mem in retrieved_memories if mem.get("memory")]
                    if memories_text_list:
                        memories_content = "\n".join(memories_text_list)
                        memory_context_message = f"Lembre-se destas interações ou informações passadas relevantes (de mem0):\n{memories_content}"
                        contexto_chat_ia.insert(0, {"role": "system", "content": memory_context_message})
            except Exception as e_mem0_search:
                st.warning(f"Aviso: Não foi possível buscar memórias com mem0: {e_mem0_search}")

        mensagens_formatadas_ia = []
        for msg_hist_ia in st.session_state["chat_principal_history"]:
            if msg_hist_ia.get("role") and msg_hist_ia.get("content"):
                mensagens_formatadas_ia.append({"role": msg_hist_ia["role"], "content": msg_hist_ia["content"]})
        contexto_chat_ia.extend(mensagens_formatadas_ia)

        if st.session_state.get("faiss_index") and st.session_state["faiss_index"].ntotal > 0 and st.session_state.get("doc_chunks"):
            try:
                client_openai_faiss = OpenAI(api_key=openai_api_key)
                query_embedding_response = client_openai_faiss.embeddings.create(input=prompt_principal, model="text-embedding-ada-002")
                query_embedding = np.array(query_embedding_response.data[0].embedding, dtype=np.float32).reshape(1, -1)
                
                D, I = st.session_state["faiss_index"].search(query_embedding, k=3)
                
                retrieved_chunks_content = ""
                if I[0][0] != -1: # Verifica se algum resultado foi encontrado
                    for idx_faiss in I[0]:
                        if idx_faiss != -1 and idx_faiss < len(st.session_state["doc_chunks"]): # Checa se o índice é válido
                            retrieved_chunks_content += st.session_state["doc_chunks"][idx_faiss] + "\n\n"
                
                if retrieved_chunks_content:
                    system_message_faiss = f"Use as seguintes informações da base de conhecimento para responder à pergunta do usuário:\n{retrieved_chunks_content}"
                    # Inserir após a memória do mem0, se existir, ou no início
                    insert_index = 1 if (contexto_chat_ia and contexto_chat_ia[0].get("role") == "system" and "mem0" in contexto_chat_ia[0].get("content","")) else 0
                    contexto_chat_ia.insert(insert_index, {"role": "system", "content": system_message_faiss})

            except Exception as e_faiss:
                st.warning(f"Erro durante a busca FAISS: {e_faiss}")

        with st.spinner("Pensando..."):
            try:
                client_final = OpenAI(api_key=openai_api_key)
                response_final = client_final.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=contexto_chat_ia,
                    temperature=0.7,
                )
                assistant_response_final = response_final.choices[0].message.content
                
                add_message_to_session(st.session_state["current_chat_session_id"], "assistant", assistant_response_final)
                st.session_state["chat_principal_history"].append({"role": "assistant", "content": assistant_response_final})

                # Adicionar a interação ao mem0 AQUI
                # current_user_id e current_agent_id já definidos acima
                if mem0_client and current_user_id:
                    try:
                        messages_to_add_to_mem0 = [
                            {"role": "user", "content": prompt_principal},
                            {"role": "assistant", "content": assistant_response_final}
                        ]
                        mem0_client.add(
                            data=messages_to_add_to_mem0, # O SDK do mem0 usa 'data=' para as mensagens
                            user_id=current_user_id
                        )
                        # st.write("Memória adicionada ao mem0.") # Para debug
                    except Exception as e_mem0_add:
                        st.warning(f"Aviso: Não foi possível adicionar memória ao mem0: {e_mem0_add}")

                with st.chat_message("assistant"):
                    st.markdown(assistant_response_final)
                st.rerun()

            except Exception as e_ia_final:
                st.warning(f"Erro ao gerar resposta da IA: {e_ia_final}")

# Controle de Navegação Principal
if "menu_sidebar" not in st.session_state:
    st.session_state["menu_sidebar"] = "Login"

if st.session_state["menu_sidebar"] == "Login":
    pagina_login()
elif st.session_state["menu_sidebar"] == "Configuração/Chat":
    pagina_chat_assistente()
elif st.session_state["menu_sidebar"] == "Chat Principal":
    pagina_chat_principal()
else:
    pagina_login() # Default para login se estado for inválido