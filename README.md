# Hubblet AI - Infraestrutura de Assistente Inteligente

## Estrutura Inicial do Projeto

```
backend/           # Backend Python, orquestração LangGraph/LangChain
  main.py
  requirements.txt
  langchain/
  langgraph/
  faiss/
  mem0/
  docling/
frontend/          # Frontend Streamlit
  app.py
  requirements.txt
knowledge/         # Processamento e indexação de conhecimento
  sources/
  faiss_index/
memories/          # Memória persistente dos usuários (mem0)
  users/

# Arquivos de configuração e documentação
README.md          # Documentação do projeto
.env.example       # Exemplo de variáveis de ambiente
```

## Decisões Técnicas
- **Python** para backend e orquestração (LangGraph, LangChain).
- **FAISS** para armazenamento vetorial local (MVP).
- **Docling** para processamento de conhecimento.
- **mem0** para memória persistente individual dos usuários.
- **Streamlit** para frontend rápido e amigável.
- Estrutura modular para facilitar futuras migrações (ex: FAISS → OpenSearch).

## Fluxo Inicial
1. Usuário define assistente personalizado (comportamento, fontes, estilo).
2. Conhecimento é processado (Docling) e indexado (FAISS).
3. Memória do usuário é armazenada separadamente (mem0).
4. Conversas são orquestradas via LangGraph, resgatando contexto e memória.
5. Frontend permite testes e uso rápido.

## Próximos Passos
- Implementar scripts de inicialização automática.
- Garantir integração entre componentes.
- Documentar cada módulo e fluxo.

---

> Estrutura pensada para agilidade, simplicidade e fácil manutenção. Sugestões de melhorias são bem-vindas!