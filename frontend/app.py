import streamlit as st
import requests
from typing import List, Dict
from datetime import datetime
import time
import os
import json # Adicionado para salvar/carregar metadados de arquivos
import numpy as np
import faiss # Mantido aqui para uso direto se necessário, embora utils.py também o tenha
from dotenv import load_dotenv
from openai import OpenAI # Para uso direto na IA de configuração

# Importações de utils (ajustado para importação direta)
from utils import (
    get_assistentes_existentes,
    reset_session,
    inicializar_faiss,
    gerar_embeddings,
    processar_arquivos,
    inicializar_mem0,
    carregar_ou_inicializar_dados_assistente
)

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

st.set_page_config(page_title="Hubblet: Criação de Assistente", layout="wide")

st.markdown("<div style='height:3.5rem;'></div>", unsafe_allow_html=True)

# A função reset_session agora é chamada explicitamente onde o botão de logout está
# Se o botão de logout estiver em app.py, a lógica de chamada a reset_session() e st.rerun() deve estar lá.
# Exemplo de como poderia ser (se o botão estiver aqui):
# if st.sidebar.button("Logout", key="logout_btn_main"):
#     if st.session_state.get('username'):
#         reset_session() # Chama a função de utils
#         st.rerun()

# As funções get_assistentes_existentes, inicializar_faiss, gerar_embeddings, processar_arquivos, inicializar_mem0
# foram movidas para utils.py e são importadas de lá.



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
        assistente = st.selectbox("Escolha um assistente", assistentes, index=0)
        submit = st.form_submit_button("Entrar")
    if submit:
        if not username.strip():
            st.warning("Usuário é obrigatório.")
            return
        st.session_state["username"] = username.strip()
        openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        mem0_api_key = os.environ.get("MEM0_API_KEY", "")

        if assistente == "Criar novo assistente":
            st.session_state["menu_sidebar"] = "Configuração/Chat"
            st.session_state["assistente_selecionado"] = None # Indica novo assistente
            # Limpa/inicializa o estado para um novo assistente
            carregar_ou_inicializar_dados_assistente(username=st.session_state["username"], nome_assistente="", openai_api_key=openai_api_key, mem0_api_key=mem0_api_key)
            st.session_state["chat_mode"] = "criar" # Definido anteriormente, mas reforça
            st.session_state["assistente_config"] = {} # Limpa config específica
            st.session_state["instrucoes_finais"] = None
            st.session_state['config_flow_initial_message_shown'] = False # Para o fluxo de config
            st.session_state['config_flow_complete'] = False
            st.session_state['current_config_step_key'] = CONFIG_STEPS_ORDER[0]

        else: # Assistente existente selecionado
            st.session_state["assistente_selecionado"] = assistente
            st.session_state["menu_sidebar"] = "Chat Principal"
            st.session_state["chat_mode"] = "editar" # Pode ser usado para carregar no modo de edição
            # Carrega os dados do assistente selecionado
            carregar_ou_inicializar_dados_assistente(username=st.session_state["username"], nome_assistente=assistente, openai_api_key=openai_api_key, mem0_api_key=mem0_api_key)
        
        st.rerun() # Força o rerun para aplicar a mudança de página

# Página de Criação/Atualização do Assistente
def pagina_chat_assistente():
    # --- Adiciona Sidebar ---
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

    # Inicialização do estado da sessão para o fluxo de configuração
    if "assistente_config" not in st.session_state: st.session_state["assistente_config"] = {}
    if "config_chat_history" not in st.session_state: st.session_state["config_chat_history"] = [] # Alterado de chat_history
    if "instrucoes_finais" not in st.session_state: st.session_state["instrucoes_finais"] = None

    is_editing = st.session_state.get("chat_mode") == "editar" and st.session_state.get("assistente_selecionado")

    if "config_flow_complete" not in st.session_state:
        st.session_state["config_flow_complete"] = True if is_editing and st.session_state.get("instrucoes_finais") else False
    
    if "current_config_step_key" not in st.session_state:
        st.session_state["current_config_step_key"] = None if st.session_state["config_flow_complete"] else CONFIG_STEPS_ORDER[0]

    if 'config_flow_initial_message_shown' not in st.session_state:
        st.session_state['config_flow_initial_message_shown'] = False

    # Mensagem inicial para modo de edição ou boas-vindas para novo assistente
    if not st.session_state['config_flow_initial_message_shown']:
        if is_editing and st.session_state.get("instrucoes_finais"):
            st.session_state["config_flow_complete"] = True 
            st.session_state["current_config_step_key"] = None
            edit_message = f"Editando '{st.session_state['assistente_selecionado']}'. As instruções atuais estão carregadas. Você pode refiná-las ou pedir para gerar uma nova versão."
            if not st.session_state["config_chat_history"] or st.session_state["config_chat_history"][-1]["content"] != edit_message: # Alterado de chat_history
                st.session_state["config_chat_history"].append({"role": "assistant", "content": edit_message}) # Alterado de chat_history
        elif not is_editing: # Novo assistente
            # O Assistente Hubblet faz a primeira pergunta diretamente na mensagem de boas-vindas.
            # Garante que o passo atual seja 'nome' para que a resposta do usuário seja corretamente atribuída.
            st.session_state['current_config_step_key'] = CONFIG_STEPS_ORDER[0]
            welcome_message = f"Olá! Eu sou o Assistente de Configuração da Hubblet e estou aqui para te ajudar a criar seu novo assistente personalizado. Vamos começar com algumas perguntas para definir a base dele. A primeira é: {CONFIG_QUESTIONS[CONFIG_STEPS_ORDER[0]]}"
            st.session_state["config_chat_history"].append({"role": "assistant", "content": welcome_message}) # Alterado de chat_history
        st.session_state['config_flow_initial_message_shown'] = True
    
    # Lógica do fluxo de configuração sequencial para NOVOS assistentes
    if not is_editing and not st.session_state["config_flow_complete"]:
        next_step_key = None
        for step_key in CONFIG_STEPS_ORDER:
            if step_key not in st.session_state["assistente_config"]:
                next_step_key = step_key
                break
        
        if next_step_key is None: # Todos os passos foram preenchidos
            st.session_state["config_flow_complete"] = True
            st.session_state["current_config_step_key"] = None
            completion_message = "Configuração básica completa! Agora você pode refinar os detalhes com a IA ou pedir para gerar as instruções finais. O que gostaria de fazer?"
            if not st.session_state["config_chat_history"] or st.session_state["config_chat_history"][-1]["content"] != completion_message: # Alterado de chat_history
                st.session_state["config_chat_history"].append({"role": "assistant", "content": completion_message}) # Alterado de chat_history
        else:
            st.session_state["current_config_step_key"] = next_step_key
            question_to_ask = CONFIG_QUESTIONS[st.session_state["current_config_step_key"]]
            # Adiciona a pergunta apenas se não for a última mensagem ou se a última foi do usuário
            if not st.session_state["config_chat_history"] or st.session_state["config_chat_history"][-1]["content"] != question_to_ask or st.session_state["config_chat_history"][-1]["role"] == "user": # Alterado de chat_history
                st.session_state["config_chat_history"].append({"role": "assistant", "content": question_to_ask}) # Alterado de chat_history

    with col_chat:
        st.markdown("<div style='font-size:1.3rem;font-weight:600;margin-bottom:0.5rem;'>Chat de Configuração</div>", unsafe_allow_html=True)
        if not openai_api_key:
            st.warning("⚠️ Defina a variável de ambiente OPENAI_API_KEY para habilitar o chat com a IA.")
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state["config_chat_history"]: # Alterado de chat_history
                cor = "#e3f2fd" if msg["role"] == "assistant" else "#e8f5e9"
                alinhamento = "flex-start" if msg["role"] == "assistant" else "flex-end"
                cor_texto = "#1976d2" if msg["role"] == "assistant" else "#388e3c"
                st.markdown(f"""
                    <div style='display:flex;justify-content:{alinhamento};margin-bottom:0.2rem;'>
                        <div style='background:{cor};color:{cor_texto};padding:0.7rem 1rem;border-radius:18px;max-width:70%;box-shadow:0 1px 2px #0001;font-size:1.05rem;'>
                            {msg['content']}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        # Mensagem inicial do assistente Hubblet
        if not st.session_state["chat_history"]:
            st.session_state["chat_history"].append({"role": "assistant", "content": "Olá! Sou o Assistente de Configuração da Hubblet. Vamos criar seu novo assistente juntos. Qual será o nome do seu assistente?"})


        with st.form(key="chat_form", clear_on_submit=True):
            prompt_placeholder = "Sua resposta..." if not st.session_state["config_flow_complete"] else "Refine os detalhes ou peça para gerar as instruções..."
            prompt = st.text_input("", placeholder=prompt_placeholder)
            enviar = st.form_submit_button("Enviar", help="Enviar mensagem", use_container_width=True)

        if enviar and prompt.strip():
            st.session_state["chat_history"].append({"role": "user", "content": prompt.strip()})

            if not st.session_state["config_flow_complete"] and st.session_state["current_config_step_key"]:
                current_key = st.session_state["current_config_step_key"]
                st.session_state["assistente_config"][current_key] = prompt.strip()
                if current_key == "nome": 
                    st.session_state["assistente_config"]["nome"] = prompt.strip()
                # A lógica para avançar para o próximo passo ou completar o fluxo está no início da função e será executada no rerun.
            else: # Fluxo de configuração completo ou modo de edição, interagir com IA
                if not openai_api_key:
                    st.session_state["chat_history"].append({"role": "assistant", "content": "OPENAI_API_KEY não configurada. Não posso processar este pedido."})
                else:
                    try:
                        from openai import OpenAI
                        client = OpenAI(api_key=openai_api_key)
                        
                        doc_context = ""
                        # Verifica se faiss_index existe e tem itens antes de tentar usá-lo
                        if st.session_state.get("faiss_index") and hasattr(st.session_state["faiss_index"], 'ntotal') and st.session_state["faiss_index"].ntotal > 0 and st.session_state.get("doc_chunks"):
                            try:
                                client_emb = OpenAI(api_key=openai_api_key)
                                user_query_emb = client_emb.embeddings.create(input=prompt.strip(), model="text-embedding-ada-002")
                                query_emb = np.array(user_query_emb.data[0].embedding, dtype=np.float32)
                                index = st.session_state["faiss_index"]
                                k = min(5, len(st.session_state["doc_chunks"]))
                                if k > 0: # Garante que k é positivo
                                    D, I = index.search(np.expand_dims(query_emb, axis=0), k)
                                    trechos_relevantes = [st.session_state["doc_chunks"][i] for i in I[0] if i < len(st.session_state["doc_chunks"])]
                                    if trechos_relevantes:
                                        doc_context = "\n\n---\nTrechos relevantes dos documentos para contexto adicional:\n" + "\n".join(trechos_relevantes)
                            except Exception as e:
                                st.warning(f"Erro ao recuperar contexto RAG: {e}")

                        config_summary_parts = []
                        if st.session_state.get("assistente_config"):
                            for key, value in st.session_state["assistente_config"].items():
                                if value: # Adiciona apenas se houver valor
                                    config_summary_parts.append(f"- {key.capitalize().replace('_', ' ')}: {value}")
                        config_summary = "\n".join(config_summary_parts)
                        
                        historico_chat_recente = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state["chat_history"][-10:]]) # Últimas 10 mensagens
                        
                        prompt_ia_parts = [] 
                        
                        if is_editing:
                            prompt_ia_parts.append("CONTEXTO ATUAL: O usuário está no modo de EDIÇÃO de um assistente existente. As instruções atuais (se houver) e o resumo da configuração estão detalhados abaixo. Seu objetivo é ajudar o usuário a refinar essas instruções ou gerar uma nova versão com base na conversa.")
                        elif st.session_state["config_flow_complete"]: # Novo assistente, configuração básica feita
                             prompt_ia_parts.append("CONTEXTO ATUAL: O usuário está criando um NOVO assistente. As respostas para as perguntas básicas de configuração já foram coletadas (resumo abaixo). Agora, sua tarefa é conversar com o usuário para aprofundar e refinar os detalhes do assistente. Explore aspectos como propósito, público-alvo, tom específico, funcionalidades desejadas, etc. Quando estiverem prontos, gere as 'Instruções Finais'.")

                        if config_summary:
                            prompt_ia_parts.append(f"\nRESUMO DA CONFIGURAÇÃO (respostas do usuário às perguntas básicas):\n{config_summary}")
                        
                        # Adiciona instruções atuais se estiver editando e elas existirem
                        if is_editing and st.session_state.get("instrucoes_finais"):
                             prompt_ia_parts.append(f"\nInstruções Finais Atuais (para edição/refinamento):\n{st.session_state['instrucoes_finais']}")
                        
                        prompt_ia_parts.append(f"\nHistórico recente da conversa:\n{historico_chat_recente}")
                        if doc_context:
                            prompt_ia_parts.append(doc_context)
                        
                        prompt_ia_parts.append("\nCom base nisso, continue a conversa para refinar detalhes ou, se solicitado ou parecer apropriado, gere as INSTRUÇÕES FINAIS completas e atualizadas para o assistente. Se precisar de mais clareza em algum ponto da configuração antes de gerar as instruções, faça perguntas objetivas. Ao gerar as instruções, comece a resposta EXATAMENTE com 'Instruções Finais:'.")
                        prompt_ia = "\n".join(prompt_ia_parts)

                        chat_resp = client.chat.completions.create(
                            model="gpt-4o", # ou gpt-3.5-turbo para mais rapidez/custo menor
                            messages=[
                                {"role": "system", "content": "Você é o Assistente de Configuração da Hubblet. Sua principal responsabilidade é guiar os usuários de forma interativa e amigável durante o processo de criação de seus próprios assistentes de IA. Siga estas diretrizes:\n1. A interface já iniciou a conversa e pode ter feito a primeira pergunta de configuração (sobre o nome do assistente). Seu papel é continuar a partir daí.\n2. Se as perguntas básicas de configuração (estilo de comunicação, funções principais, fontes de informação/documentos) ainda não foram todas respondidas pelo usuário (verifique o histórico da conversa e o resumo da configuração fornecidos), continue a fazê-las de forma sequencial e clara, uma por vez. Use as perguntas pré-definidas como referência para o que perguntar.\n3. Após o usuário ter respondido às perguntas básicas de configuração, seu objetivo é conversar com ele para refinar os detalhes. Ajude-o a pensar sobre o propósito do assistente, o público-alvo, o tom desejado, e quaisquer funcionalidades específicas. Ofereça sugestões e melhores práticas quando apropriado.\n4. Seja proativo em pedir esclarecimentos se as respostas do usuário forem vagas ou incompletas.\n5. Quando você julgar que tem informações suficientes e o usuário parece satisfeito com a definição do assistente, ou se o usuário explicitamente pedir para gerar as instruções, crie as 'Instruções Finais'. Estas instruções devem ser um prompt completo e bem estruturado, pronto para ser usado como o system prompt do novo assistente do cliente.\n6. Ao gerar as 'Instruções Finais', sua resposta DEVE começar EXATAMENTE com a frase 'Instruções Finais:'. Não adicione nenhum texto antes disso nessa resposta específica.\n7. Mantenha um tom prestativo, paciente e profissional durante toda a interação. Lembre-se que o usuário pode não ser técnico."},
                                {"role": "user", "content": prompt_ia}
                            ],
                            temperature=0.3
                        )
                        resposta_ia = chat_resp.choices[0].message.content.strip()
                        st.session_state["chat_history"].append({"role": "assistant", "content": resposta_ia})
                        if resposta_ia.lower().startswith("instruções finais:"):
                            st.session_state["instrucoes_finais"] = resposta_ia # Armazena incluindo o prefixo
                    except Exception as e:
                        st.session_state["chat_history"].append({"role": "assistant", "content": f"Erro ao gerar resposta da IA: {e}"})
            st.rerun()

        if st.session_state.get("instrucoes_finais"):
            st.success("Instruções finais do assistente prontas! Clique em Salvar para finalizar.")
    with col_upload:
        st.markdown("<div style='font-size:1.3rem;font-weight:600;margin-bottom:0.5rem;'>Documentos do Assistente</div>", unsafe_allow_html=True)
        st.markdown("<div style='color:#666;font-size:1rem;margin-bottom:0.7rem;'>Anexe documentos que seu assistente poderá usar.</div>", unsafe_allow_html=True)
        arquivos = st.file_uploader("Arraste ou clique para selecionar arquivos", type=["pdf", "txt", "csv", "docx"], accept_multiple_files=True, key="file_uploader")
        if arquivos:
            # Processa os arquivos e obtém chunks, embeddings e nomes dos arquivos processados
            novos_doc_chunks, novos_embeddings, nomes_arquivos_processados = processar_arquivos(arquivos, openai_api_key)
            
            if nomes_arquivos_processados: # Apenas atualiza se arquivos foram processados com sucesso
                # Adiciona os nomes dos novos arquivos à lista existente (evita duplicatas se o mesmo arquivo for reenviado)
                current_uploaded_files = st.session_state.get("uploaded_files", [])
                for nome_arq in nomes_arquivos_processados:
                    if nome_arq not in current_uploaded_files:
                        current_uploaded_files.append(nome_arq)
                st.session_state["uploaded_files"] = current_uploaded_files
                
                st.session_state["doc_chunks"].extend(novos_doc_chunks)
                
                if "faiss_index" not in st.session_state or st.session_state["faiss_index"] is None:
                    st.session_state["faiss_index"] = inicializar_faiss()
                
                index = st.session_state["faiss_index"]
                if novos_embeddings: 
                    for emb in novos_embeddings:
                        index.add(np.expand_dims(emb, axis=0))
                st.success(f"{len(nomes_arquivos_processados)} arquivo(s) processado(s) e adicionado(s) ao contexto do assistente.")
                # Limpa o file_uploader para permitir novos uploads sem manter os antigos na interface do uploader
                # st.session_state["file_uploader"] = [] # Isso pode causar problemas se a chave não for gerenciada corretamente
                # A melhor abordagem é usar uma chave diferente para o file_uploader se precisar resetá-lo ou usar st.empty()
            elif arquivos: # Se arquivos foram fornecidos mas nenhum foi processado com sucesso
                st.warning("Nenhum dos arquivos fornecidos pôde ser processado ou adicionado.")
            # O salvamento do índice, chunks e lista de nomes de arquivos será feito no botão "Salvar Assistente"
        if st.session_state.get("uploaded_files"):
            st.markdown("<div style='margin-top:0.5rem;'><b>Arquivos no contexto do assistente:</b></div>", unsafe_allow_html=True)
            # uploaded_files agora é uma lista de nomes de arquivos
            for nome_arquivo in st.session_state["uploaded_files"]:
                # Não temos mais o objeto UploadedFile aqui para pegar o tamanho facilmente após o processamento inicial.
                # Poderíamos armazenar metadados (nome, tamanho) se necessário, ou apenas listar os nomes.
                st.markdown(f"<div style='padding:0.3rem 0.5rem;background:#f7f7f7;border-radius:8px;margin-bottom:0.2rem;font-size:0.97rem;'><span>{nome_arquivo}</span><span style='color:#388e3c; float:right;'>Processado</span></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='color:#aaa;font-size:0.95rem;'>Nenhum arquivo enviado ainda.</div>", unsafe_allow_html=True)
    st.markdown("<div style='margin-top:2rem;display:flex;justify-content:center;'>", unsafe_allow_html=True)
    nome_assistente_config = st.session_state.get("assistente_config", {}).get("nome", "").strip()
    nome_assistente_selecionado = (st.session_state.get("assistente_selecionado") or "").strip()

    # Prioriza o nome da configuração, depois o selecionado (para edição)
    nome_assistente_para_salvar = nome_assistente_config if nome_assistente_config else nome_assistente_selecionado
    
    salvar_label = "Atualizar Assistente" if nome_assistente_selecionado and nome_assistente_selecionado == nome_assistente_para_salvar else "Salvar Novo Assistente"
    if not nome_assistente_para_salvar and st.session_state.get("chat_mode") == "criar": # Se criando e nome ainda não definido
        salvar_label = "Salvar Assistente"

    habilitar_salvar = bool(st.session_state.get("instrucoes_finais")) and bool(nome_assistente_para_salvar)

    if st.button(f"✅ {salvar_label}", key="salvar_btn", use_container_width=True, disabled=not habilitar_salvar):
        instrucoes = st.session_state.get("instrucoes_finais", "")
        mem0_api_key = os.environ.get("MEM0_API_KEY")
        
        if instrucoes and nome_assistente_para_salvar:
            safe_nome_assistente = nome_assistente_para_salvar.replace(' ', '_').lower().strip()
            # Define o diretório base para salvar os dados do assistente
            # app.py está em frontend/, então o diretório de assistentes deve ser um subdiretório de frontend.
            path_base = os.path.join(os.path.dirname(__file__), "assistentes_salvos")
            os.makedirs(path_base, exist_ok=True) # Garante que o diretório exista

            # Salvar instruções
            instrucoes_file = os.path.join(path_base, f"assistente_{safe_nome_assistente}_instrucoes.txt")
            with open(instrucoes_file, "w", encoding="utf-8") as f_inst:
                f_inst.write(instrucoes)
            
            # Salvar FAISS index
            faiss_file = os.path.join(path_base, f"assistente_{safe_nome_assistente}_faiss.index")
            if st.session_state.get("faiss_index") and st.session_state["faiss_index"].ntotal > 0:
                try:
                    faiss.write_index(st.session_state["faiss_index"], faiss_file)
                except Exception as e:
                    st.error(f"Erro ao salvar índice FAISS: {e}")
            elif os.path.exists(faiss_file): # Se não há índice na sessão mas existe arquivo, remover para evitar inconsistência
                try: os.unlink(faiss_file)
                except: pass
            
            # Salvar doc_chunks
            chunks_file = os.path.join(path_base, f"assistente_{safe_nome_assistente}_chunks.json")
            if st.session_state.get("doc_chunks"):
                with open(chunks_file, "w", encoding="utf-8") as f_chunks:
                    json.dump(st.session_state["doc_chunks"], f_chunks)
            elif os.path.exists(chunks_file):
                try: os.unlink(chunks_file)
                except: pass

            # Salvar nomes dos arquivos originais (uploaded_files)
            uploaded_files_info_file = os.path.join(path_base, f"assistente_{safe_nome_assistente}_uploaded_files.json")
            if st.session_state.get("uploaded_files"):
                with open(uploaded_files_info_file, "w", encoding="utf-8") as f_uploaded_info:
                    json.dump(st.session_state.get("uploaded_files", []), f_uploaded_info)
            elif os.path.exists(uploaded_files_info_file):
                try: os.unlink(uploaded_files_info_file)
                except: pass

            st.session_state["mem0_instance"] = inicializar_mem0(nome_assistente_para_salvar, mem0_api_key)
            
            st.session_state["assistente_selecionado"] = nome_assistente_para_salvar
            st.session_state["assistente_config"]["nome"] = nome_assistente_para_salvar
            st.session_state["chat_mode"] = "editar" # Após salvar, estamos efetivamente no modo de edição

            if st.session_state["mem0_instance"]:
                st.success(f"Assistente '{nome_assistente_para_salvar}' salvo/atualizado e memória inicializada!")
            else:
                st.warning(f"Assistente '{nome_assistente_para_salvar}' salvo/atualizado, mas erro ao inicializar Mem0. Verifique a chave API.")
            
            # Forçar um rerun para atualizar a lista de assistentes na sidebar, se aplicável, e estado geral.
            st.rerun()

        elif not nome_assistente_para_salvar:
            st.warning("Por favor, defina um nome para o assistente no chat de configuração antes de salvar.")
        else: # Sem instruções
            st.warning("Gere as instruções finais através do chat de configuração antes de salvar.")
    st.markdown("</div>", unsafe_allow_html=True)
    if st.session_state.get("assistente_selecionado"):
        st.markdown("---")
        st.subheader("Status do Assistente")
        st.info(f"Última atualização: {datetime.now().strftime('%d/%m/%Y %H:%M')} | Fontes anexadas: {len(st.session_state.get('uploaded_files', []))}")

# Página de Chat Principal
def pagina_chat_principal():
    st.title(f"Chat com {st.session_state.get('assistente_selecionado', 'Assistente')}")

    # --- Adiciona Sidebar para Navegação e Histórico --- 
    with st.sidebar:
        st.title(f"Hubblet AI")
        st.write(f"Usuário: **{st.session_state.get('username','')}**")
        st.write(f"Assistente: **{st.session_state.get('assistente_selecionado','N/A')}**")
        st.divider()
        st.subheader("Navegação")
        if st.button("Configurar/Trocar Assistente", key="goto_config_btn_chat_main"):
            st.session_state["menu_sidebar"] = "Login" # Volta para a seleção
            st.rerun()
        if st.button("Novo Chat", key="novo_chat_btn"):
            if "chat_principal_history" in st.session_state and st.session_state["chat_principal_history"]:
                # Salvar histórico atual no Mem0 antes de limpar
                if st.session_state.get("mem0_instance"):
                    try:
                        for msg in st.session_state["chat_principal_history"]:
                            st.session_state["mem0_instance"].add(msg["content"], user_id=st.session_state["mem0_instance"].user_id, role=msg["role"])
                        st.success("Histórico do chat atual salvo no Mem0.")
                    except Exception as e:
                        st.error(f"Erro ao salvar histórico do chat no Mem0: {e}")

            st.session_state["chat_principal_history"] = []
            st.rerun()
        st.divider()
        st.subheader("Histórico da Conversa")
        historico_mem0_container = st.container()

    # Layout principal: Chat à esquerda, (opcional) informações adicionais à direita
    col_chat_principal, col_historico_lateral = st.columns([2, 1])

    with col_historico_lateral:
        st.markdown("<div style='font-size:1.2rem;font-weight:600;margin-bottom:0.5rem;'>Histórico (Mem0)</div>", unsafe_allow_html=True)
        mem0_history_display = st.empty() # Placeholder para exibir o histórico do Mem0

    # Exibir histórico do chat carregado na sidebar
    historico_para_exibir = st.session_state.get("chat_history", [])
    if historico_para_exibir:
        for msg in historico_para_exibir:
            role = "Usuário" if msg["role"] == "user" else "Assistente"
            mem0_history_display.markdown(f"**{role}:** {msg['content']}")
        # Adiciona um separador visual
        mem0_history_display.markdown("--- Memória Persistida ---")
    else:
        mem0_history_display.info("Histórico do chat vazio.")

    # Layout principal: Chat à esquerda, (opcional) informações adicionais à direita
    col_chat_principal, col_historico_lateral = st.columns([2, 1])

    with col_historico_lateral:
        st.markdown("<div style='font-size:1.2rem;font-weight:600;margin-bottom:0.5rem;'>Histórico (Mem0)</div>", unsafe_allow_html=True)
        mem0_history_display = st.empty() # Placeholder para exibir o histórico do Mem0

    # Carregar histórico do Mem0 para exibição na coluna lateral
    historico_mem0_mensagens = []
    if st.session_state.get("mem0_instance"):
        try:
            # Mem0 get_all() retorna uma lista de objetos Memory
            raw_history = st.session_state["mem0_instance"].get_all()
            if raw_history:
                # Ordena por created_at para exibir em ordem cronológica
                for entry in sorted(raw_history, key=lambda x: x.get('created_at', '')):
                    # A estrutura do objeto Memory pode variar, verificar 'memory' ou 'text'
                    memory_data = entry.get('memory', {})
                    if 'role' in memory_data and 'content' in memory_data:
                        historico_mem0_mensagens.append({"role": memory_data['role'], "content": memory_data['content']})
                    elif 'text' in entry:
                         # Tenta inferir o papel se apenas 'text' estiver presente
                         # Verifica se o texto começa com "user: " ou "assistant: " (case-insensitive)
                         text_lower = entry['text'].lower()
                         if text_lower.startswith("user: "):
                             historico_mem0_mensagens.append({"role": "user", "content": entry['text'][6:]})
                         elif text_lower.startswith("assistant: "):
                             historico_mem0_mensagens.append({"role": "assistant", "content": entry['text'][11:]})
                         else:
                             historico_mem0_mensagens.append({"role": "system", "content": entry['text']}) # ou 'unknown'
        except Exception as e:
            st.sidebar.error(f"Erro ao carregar histórico do Mem0: {e}")

    # Exibir histórico do Mem0 na coluna lateral
    with mem0_history_display.container():
        if historico_mem0_mensagens:
            for msg in historico_mem0_mensagens:
                st.markdown(f"**{msg['role'].capitalize()}:** {msg['content']}")
        else:
            st.info("Nenhum histórico de conversa encontrado no Mem0.")

    with col_chat_principal:
        st.markdown("<div style='font-size:1.3rem;font-weight:600;margin-bottom:0.5rem;'>Chat Interativo</div>", unsafe_allow_html=True)
        
        # Inicializa o histórico de chat da sessão se não existir
        if "chat_principal_history" not in st.session_state:
            st.session_state["chat_principal_history"] = [] # Este é o histórico da sessão atual

        # Exibir mensagens do chat da sessão atual
        chat_display_container = st.container()
        with chat_display_container:
            for msg in st.session_state["chat_principal_history"]:
                cor = "#e3f2fd" if msg["role"] == "assistant" else "#e8f5e9"
                alinhamento = "flex-start" if msg["role"] == "assistant" else "flex-end"
                cor_texto = "#1976d2" if msg["role"] == "assistant" else "#388e3c"
                st.markdown(f"""
                    <div style='display:flex;justify-content:{alinhamento};margin-bottom:0.2rem;'>
                        <div style='background:{cor};color:{cor_texto};padding:0.7rem 1rem;border-radius:18px;max-width:70%;box-shadow:0 1px 2px #0001;font-size:1.05rem;'>
                            {msg['content']}
                        </div>
                    </div>
                """, unsafe_allow_html=True)

        # Input do usuário
        prompt_chat = st.chat_input("Digite sua mensagem...")

        if prompt_chat:
            st.session_state["chat_principal_history"].append({"role": "user", "content": prompt_chat})
            
            # Salvar mensagem do usuário no Mem0
            if st.session_state.get("mem0_instance"):
                try:
                    st.session_state["mem0_instance"].add(prompt_chat, user_id=st.session_state["mem0_instance"].user_id, role="user")
                except Exception as e:
                    st.error(f"Erro ao salvar mensagem do usuário no Mem0: {e}")
            
            # Lógica para obter resposta do assistente (exemplo)
            # Aqui você integraria com LangGraph ou diretamente com a OpenAI API usando as instruções do assistente
            openai_api_key = os.environ.get("OPENAI_API_KEY", "")
            if not openai_api_key:
                resposta_assistente = "OPENAI_API_KEY não configurada. Não posso responder."
            elif not st.session_state.get("instrucoes_finais"):
                resposta_assistente = "Instruções do assistente não carregadas. Configure o assistente primeiro."
            else:
                try:
                    client = OpenAI(api_key=openai_api_key)
                    
                    # Construir histórico para a API da OpenAI a partir do chat_principal_history
                    messages_for_api = [
                        {"role": "system", "content": st.session_state["instrucoes_finais"]}
                    ]
                    for msg in st.session_state["chat_principal_history"][-10:]: # Últimas 10 mensagens da sessão
                        messages_for_api.append(msg)

                    # Adicionar contexto RAG se houver
                    doc_context_rag = ""
                    if st.session_state.get("faiss_index") and hasattr(st.session_state["faiss_index"], 'ntotal') and st.session_state["faiss_index"].ntotal > 0 and st.session_state.get("doc_chunks"):
                        try:
                            client_emb = OpenAI(api_key=openai_api_key)
                            user_query_emb = client_emb.embeddings.create(input=prompt_chat, model="text-embedding-ada-002")
                            query_emb_np = np.array(user_query_emb.data[0].embedding, dtype=np.float32)
                            index_faiss = st.session_state["faiss_index"]
                            k_rag = min(3, len(st.session_state["doc_chunks"])) # Top 3 chunks
                            if k_rag > 0:
                                D_rag, I_rag = index_faiss.search(np.expand_dims(query_emb_np, axis=0), k_rag)
                                trechos_relevantes_rag = [st.session_state["doc_chunks"][i] for i in I_rag[0] if i < len(st.session_state["doc_chunks"])]
                                if trechos_relevantes_rag:
                                    doc_context_rag = "\n\nContexto dos documentos:\n" + "\n".join(trechos_relevantes_rag)
                                    # Adicionar ao system prompt ou como uma mensagem de system antes do user prompt
                                    messages_for_api[0]["content"] += doc_context_rag # Adiciona ao system prompt
                        except Exception as e_rag:
                            st.warning(f"Erro ao buscar contexto RAG no chat principal: {e_rag}")

                    # Adicionar memória do Mem0 como contexto (últimas interações)
                    mem0_context = ""
                    if historico_mem0_mensagens: # Usa o histórico já carregado
                        mem0_context_list = []
                        for mem_msg in historico_mem0_mensagens[-5:]: # Últimas 5 interações do Mem0
                            mem0_context_list.append(f"{mem_msg['role']}: {mem_msg['content']}")
                        if mem0_context_list:
                            mem0_context = "\n\nLembre-se destas interações anteriores:\n" + "\n".join(mem0_context_list)
                            messages_for_api[0]["content"] += mem0_context # Adiciona ao system prompt

                    completion = client.chat.completions.create(
                        model="gpt-4o", # ou o modelo configurado
                        messages=messages_for_api,
                        temperature=0.7
                    )
                    resposta_assistente = completion.choices[0].message.content.strip()
                except Exception as e:
                    resposta_assistente = f"Erro ao contatar a IA: {e}"

            st.session_state["chat_principal_history"].append({"role": "assistant", "content": resposta_assistente})

            # Salvar resposta do assistente no Mem0
            if st.session_state.get("mem0_instance"):
                try:
                    st.session_state["mem0_instance"].add(resposta_assistente, user_id=st.session_state["mem0_instance"].user_id, role="assistant")
                except Exception as e:
                    st.error(f"Erro ao salvar resposta do assistente no Mem0: {e}")
            
            st.rerun() # Atualiza a interface com a nova mensagem

# --- Roteamento de Páginas --- (Baseado no estado da sessão)
if "menu_sidebar" not in st.session_state:
    st.session_state["menu_sidebar"] = "Login"

# Define a página inicial como Login/Assistente
if st.session_state["menu_sidebar"] == "Login/Assistente":
    pagina_login()
elif st.session_state["menu_sidebar"] == "Configuração/Chat":
    # Verifica se está logado para acessar a configuração
    if "username" not in st.session_state:
        st.session_state["menu_sidebar"] = "Login/Assistente" # Redireciona para login
        st.rerun()
    else:
        pagina_chat_assistente()
elif st.session_state["menu_sidebar"] == "Chat Principal":
    # Verifica se está logado para acessar o chat principal
    if "username" not in st.session_state:
        st.session_state["menu_sidebar"] = "Login/Assistente" # Redireciona para login
        st.rerun()
    else:
        pagina_chat_principal()
else:
    # Fallback: Se o estado for inválido, volta para o login
    st.session_state["menu_sidebar"] = "Login/Assistente"
    pagina_login()
    st.rerun()