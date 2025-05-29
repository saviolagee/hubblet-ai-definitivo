# Módulo para construir o grafo de conversação usando LangGraph

from typing import TypedDict, Annotated, Sequence
import operator
from langgraph.graph import StateGraph, END
from mem0 import MemoryClient
from src.data_persistence.faiss.faiss_retriever import search_knowledge, DIMENSION
from openai import OpenAI
import numpy as np
import os

# 1. Definir o Estado do Grafo
class AgentState(TypedDict):
    user_input: str
    memory_context: str
    knowledge_context: str
    response: str

# 2. Definir os Nós do Grafo (Funções reais)
def retrieve_memory(state: AgentState) -> AgentState:
    print("---NÓ: RECUPERAR MEMÓRIA---")
    user_input = state['user_input']
    user_id = os.environ.get("USER_ID", "default_user") # Or however you get the user_id for the graph
    agent_id = os.environ.get("AGENT_ID", "graph_agent") # Or however you get the agent_id
    mem0_api_key = os.environ.get("MEM0_API_KEY")
    if not mem0_api_key:
        print("AVISO: MEM0_API_KEY não configurada. Testes de memória podem falhar.")
        mem0_client = MemoryClient()
    else:
        mem0_client = MemoryClient(api_key=mem0_api_key)
    
    memory_results = mem0_client.search(user_input, user_id=user_id, agent_id=agent_id)
    # Format results for context; this might need adjustment based on mem0_client.search() output
    if memory_results:
        memory_context = "\n".join([res.get('text', '') for res in memory_results])
    else:
        memory_context = "Nenhuma memória relevante encontrada."
    print(f"Contexto recuperado da memória: {memory_context}")
    return {"memory_context": memory_context}

def retrieve_knowledge(state: AgentState) -> AgentState:
    print("---NÓ: RECUPERAR CONHECIMENTO---")
    user_input = state['user_input']
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY não definido.")
    client = OpenAI(api_key=openai_api_key)
    embedding_resp = client.embeddings.create(input=user_input, model="text-embedding-ada-002")
    query_vector = np.array(embedding_resp.data[0].embedding, dtype=np.float32)
    distances, indices = search_knowledge(query_vector, k=3)
    knowledge_context = f"Índices encontrados: {indices}, Distâncias: {distances}"
    print(f"Contexto recuperado do conhecimento: {knowledge_context}")
    return {"knowledge_context": knowledge_context}

def generate_response(state: AgentState) -> AgentState:
    print("---NÓ: GERAR RESPOSTA---")
    user_input = state['user_input']
    memory_context = state['memory_context']
    knowledge_context = state['knowledge_context']
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY não definido.")
    client = OpenAI(api_key=openai_api_key)
    system_prompt = "Você é um assistente inteligente que responde de forma clara e objetiva, usando contexto de memória e conhecimento."
    prompt = f"Usuário: {user_input}\nMemória: {memory_context}\nConhecimento: {knowledge_context}"
    chat_resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
        temperature=0.2
    )
    response = chat_resp.choices[0].message.content.strip()
    print(f"Resposta gerada: {response}")
    return {"response": response}

# 2. Definir os Nós do Grafo (Funções reais)
def ia_configuradora(state: AgentState) -> AgentState:
    print("---NÓ: IA CONFIGURADORA---")
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY não definido.")
    client = OpenAI(api_key=openai_api_key)
    historico = state.get('memory_context', '')
    prompt = f"Você é uma IA especialista em criar assistentes personalizados. Com base no histórico: {historico}, faça a próxima pergunta para configurar um novo assistente. Se todas as informações já foram coletadas, gere as instruções finais do assistente." 
    chat_resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "Conduza o usuário na configuração do assistente, perguntando apenas o necessário."}, {"role": "user", "content": prompt}],
        temperature=0.2
    )
    proxima_pergunta = chat_resp.choices[0].message.content.strip()
    print(f"Pergunta/instrução gerada: {proxima_pergunta}")
    return {"response": proxima_pergunta}

# 3. Construir o Grafo
workflow = StateGraph(AgentState)
workflow.add_node("ia_configuradora", ia_configuradora)
workflow.add_node("memoria", retrieve_memory)
workflow.add_node("conhecimento", retrieve_knowledge)
workflow.add_node("resposta", generate_response)
workflow.add_edge("ia_configuradora", "memoria")
workflow.add_edge("memoria", "conhecimento")
workflow.add_edge("conhecimento", "resposta")
workflow.add_edge("resposta", END)
workflow.set_entry_point("ia_configuradora")

def run_graph(user_input: str) -> str:
    state = {"user_input": user_input, "memory_context": "", "knowledge_context": "", "response": ""}
    result = workflow.run(state)
    return result["response"]

if __name__ == "__main__":
    print("Testando execução do LangGraph...")
    test_input = "Qual a relação entre gatos e felinos?"
    resposta = run_graph(test_input)
    print(f"Resposta final: {resposta}")