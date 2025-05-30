# Hubblet AI - Sistema de Chatbot com Memória e Base de Conhecimento

## 1. Visão Geral (Para Leigos)

O Hubblet AI é uma plataforma que permite criar e interagir com assistentes de inteligência artificial (chatbots) personalizados.

**O que ele faz?**
Você pode construir diferentes "personalidades" de IA, cada uma com seu próprio nome, estilo de conversa e conhecimento específico. Por exemplo, você pode ter um assistente especialista em culinária e outro que te ajuda a planejar viagens.

**Para quem é?**
Para qualquer pessoa que queira:
*   Criar chatbots para tarefas específicas.
*   Ter conversas mais ricas e contextuais com uma IA, pois o Hubblet AI lembra de interações passadas.
*   Alimentar a IA com seus próprios documentos (PDFs, textos) para que ela possa responder perguntas baseadas nesse material.

**Principais Funcionalidades:**
*   **Criação de Múltiplos Assistentes:** Defina nome, estilo de comunicação e instruções específicas para cada assistente.
*   **Memória de Longo Prazo:** Os assistentes lembram de conversas anteriores com você (usando a API Mem0).
*   **Base de Conhecimento Personalizada:** Faça upload de arquivos (.txt, .md, .pdf) para que o assistente use essas informações como referência em suas respostas (usando FAISS e OpenAI Embeddings).
*   **Interface Amigável:** Um sistema de chat fácil de usar construído com Streamlit.
*   **Login de Usuário:** Suas configurações e assistentes são salvos por usuário.

## 2. Como Usar (Fluxo Típico)

1.  **Login:**
    *   Acesse a aplicação.
    *   Digite um nome de usuário. Isso ajuda a manter seus assistentes e conversas separados.
2.  **Seleção ou Criação de Assistente:**
    *   Na tela de login, você pode escolher um assistente já existente na lista ou selecionar "Criar novo assistente".
3.  **Configuração do Assistente (se novo ou editando):**
    *   Você será guiado por um chat de configuração para definir:
        *   **Nome:** Um nome para seu assistente.
        *   **Estilo:** Como ele deve se comunicar (formal, amigável, etc.).
        *   **Funções:** O que ele deve fazer.
        *   **Fontes de Informação:** Confirmação sobre o upload de documentos.
    *   **Base de Conhecimento (Opcional):** Na coluna da direita, você pode fazer upload de arquivos (.txt, .md, .pdf). Esses arquivos serão processados e o assistente poderá usá-los para responder perguntas.
    *   **Instruções Finais:** O chat de configuração ajudará a gerar um conjunto de instruções que guiarão o comportamento do assistente.
    *   **Salvar:** Após configurar, clique em "Salvar Assistente".
4.  **Interagindo no Chat Principal:**
    *   Após selecionar ou salvar um assistente, você será direcionado para a tela de chat principal.
    *   Digite suas perguntas ou comandos.
    *   O assistente usará suas instruções, a memória de conversas anteriores (Mem0) e a base de conhecimento (FAISS) para responder.
5.  **Gerenciamento de Conversas:**
    *   Na barra lateral, você pode ver "Minhas Conversas" (sessões de chat anteriores com o assistente atual).
    *   Clique em uma conversa para carregar o histórico.
    *   Clique em "Nova Conversa" para iniciar um novo chat com o mesmo assistente, limpando o histórico da tela atual (mas a memória de longo prazo com Mem0 ainda é mantida).

## 3. Arquitetura e Componentes (Técnico)

O sistema é construído principalmente em Python usando a biblioteca Streamlit para a interface do usuário.

*   **Frontend (`src/frontend/app.py` e `src/frontend/utils.py`):**
    *   **`app.py`**: Contém a lógica principal da interface do usuário Streamlit, dividida em páginas:
        *   `pagina_login()`: Gerencia o login do usuário e a seleção/criação de assistentes.
        *   `pagina_chat_assistente()`: Interface para configurar novos assistentes ou editar existentes, incluindo o upload de arquivos para a base de conhecimento.
        *   `pagina_chat_principal()`: A interface de chat principal onde o usuário interage com o assistente selecionado.
    *   **`utils.py`**: Contém funções auxiliares para:
        *   Gerenciamento de estado da sessão do Streamlit (`st.session_state`).
        *   Processamento de arquivos (leitura, chunking).
        *   Geração de embeddings com OpenAI.
        *   Criação e consulta de índices FAISS.
        *   Salvamento e carregamento de configurações de assistentes e históricos de chat.
    *   **Gerenciamento de Estado da Sessão:** O Streamlit (`st.session_state`) é usado extensivamente para manter o estado da aplicação entre interações do usuário (ex: usuário logado, assistente selecionado, histórico de chat da UI, etc.).

*   **Backend e Serviços:**
    *   **OpenAI API:**
        *   Usada para gerar respostas de texto dos assistentes (ex: `gpt-3.5-turbo`, `gpt-4`).
        *   Usada para gerar embeddings (vetores numéricos que representam o significado do texto) para os documentos da base de conhecimento (modelo `text-embedding-ada-002`).
        *   Requer uma `OPENAI_API_KEY` configurada como variável de ambiente.
    *   **Mem0 API:**
        *   Utilizada para fornecer memória de longo prazo aos assistentes.
        *   As interações (perguntas do usuário e respostas do assistente) são salvas e recuperadas com base no `user_id`.
        *   Permite que o assistente "lembre" de contextos de conversas anteriores.
    *   **FAISS (Facebook AI Similarity Search):**
        *   Uma biblioteca para busca eficiente de similaridade em vetores.
        *   Usada para criar um índice com os embeddings dos documentos que você carrega.
        *   Quando você faz uma pergunta, o sistema converte sua pergunta em um embedding e usa o FAISS para encontrar os trechos mais relevantes dos seus documentos para usar como contexto na resposta da IA.

*   **Gerenciamento de Dados:**
    *   **Configurações dos Assistentes:**
        *   Salvas em: `src/frontend/assistentes_salvos/<username>/<nome_assistente_seguro>/`
        *   `config.json`: Contém as instruções finais e metadados do assistente.
        *   `faiss_index.idx`: O arquivo de índice FAISS local para os documentos do assistente.
        *   `document_chunks.json`: Os trechos de texto extraídos dos documentos carregados.
        *   `<nome_assistente_seguro>` é uma versão do nome do assistente adaptada para nomes de diretório.
    *   **Histórico de Conversas (Sessões de Chat):**
        *   Salvo em: `src/chat_history.json`
        *   Este arquivo JSON contém uma lista de todas as sessões de chat de todos os usuários. Cada sessão inclui um ID, `user_id`, título, timestamps e uma lista das mensagens trocadas.
    *   **Variáveis de Ambiente:**
        *   `OPENAI_API_KEY`: Essencial para a funcionalidade da OpenAI. Pode ser definida diretamente no ambiente ou em um arquivo `.env` na raiz do projeto.

## 4. Estrutura de Arquivos e Dados Importantes

```
c:\hubblet ai\
├── .env (Opcional, para OPENAI_API_KEY)
├── requirements.txt
├── README.md (Este arquivo)
└── src\
    ├── chat_history.json (Histórico de todas as sessões de chat)
    └── frontend\
        ├── app.py (Lógica principal da UI Streamlit)
        ├── utils.py (Funções auxiliares)
        └── assistentes_salvos\
            └── <username>\
                └── <nome_assistente_seguro>\
                    ├── config.json (Instruções e metadados do assistente)
                    ├── faiss_index.idx (Índice FAISS dos documentos)
                    └── document_chunks.json (Textos dos documentos)
```

## 5. Como Executar Localmente

1.  **Pré-requisitos:**
    *   Python (versão 3.8 ou superior recomendada).
    *   `pip` (gerenciador de pacotes Python).
2.  **Clonar o Repositório (se aplicável):**
    *   Se você obteve o código como um arquivo zip, extraia-o.
    *   Se for de um repositório Git: `git clone <url_do_repositorio>`
3.  **Criar e Ativar um Ambiente Virtual (Recomendado):**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```
4.  **Instalar Dependências:**
    *   Navegue até o diretório raiz do projeto (onde `requirements.txt` está localizado).
    ```bash
    pip install -r requirements.txt
    ```
5.  **Configurar a Chave da API OpenAI:**
    *   Você precisa de uma chave da API da OpenAI.
    *   **Opção A (Variável de Ambiente):**
        Defina a variável de ambiente `OPENAI_API_KEY` no seu sistema.
    *   **Opção B (Arquivo `.env`):**
        Crie um arquivo chamado `.env` na raiz do projeto (`c:\hubblet ai\.env`) com o seguinte conteúdo:
        ```
        OPENAI_API_KEY="sua_chave_api_aqui"
        ```
6.  **Executar a Aplicação Streamlit:**
    *   No terminal, a partir do diretório raiz do projeto:
    ```bash
    streamlit run src/frontend/app.py
    ```
    *   O Streamlit geralmente abre a aplicação automaticamente no seu navegador padrão. Caso contrário, ele mostrará um endereço local (como `http://localhost:8501`) para você acessar.

## 6. Possíveis Melhorias Futuras
*   Interface de administração para gerenciar usuários e assistentes.
*   Melhorias na interface de upload e gerenciamento de documentos.
*   Suporte a mais tipos de arquivos para a base de conhecimento.
*   Opções avançadas de configuração para o Mem0 e FAISS.
*   Internacionalização (suporte a múltiplos idiomas).
*   Testes automatizados.