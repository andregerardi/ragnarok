import charset_normalizer
from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st
import pandas as pd
import json
import math
import re
import os

def extrair_json(resposta):
    """
    Extrai o JSON de uma string de resposta.
    """
    try:
        # Expressão regular para encontrar o JSON entre colchetes
        match = re.search(r'\[.*\]', resposta, re.DOTALL)
        if match:
            json_str = match.group(0)
            # Carrega o JSON em um objeto Python
            data = json.loads(json_str)
            return data
        else:
            match = re.search(r'\{.*\}', resposta, re.DOTALL)
            if match:
                json_str = match.group(0)
                # Carrega o JSON em um objeto Python
                data = json.loads(json_str)
                return [data]
            else:
                print("JSON não encontrado na resposta.")
                return None
    except json.JSONDecodeError as e:
        print(f"Erro ao decodificar JSON: {e}")
        return None

# Carrega variáveis de ambiente
load_dotenv()

# Configuração inicial do app
st.set_page_config(page_title="Gerenciador de Dados", layout="wide")

# Inicializa session_state para armazenar os dados
if "data" not in st.session_state:
    st.session_state.data = {}

if "uploaded_json" not in st.session_state:
    st.session_state.uploaded_json = None

if "csv_data" not in st.session_state:
    st.session_state.csv_data = []

if "results_df" not in st.session_state:
    st.session_state.results_df = pd.DataFrame()

# Recupera o token de autenticação
TOKEN = st.secrets["auth_token"]
# Conexão com OpenAI Databricks
client = OpenAI(
    api_key=TOKEN,
    base_url="https://fgv-pocs-genie.cloud.databricks.com/serving-endpoints"
)

# Criando abas para navegação
#tab0, tab1, tab2, tab3 = st.tabs(["🤖 Processar Documentos", "📂 Upload de Base via CSV", "📥 Cadastro via JSON", "📋 Cadastro via Formulário"])
tab0, tab1, tab3 = st.tabs(["🤖 Processar Documentos", "📂 Upload de Base via CSV", "📋 Cadastro via Formulário"])

# ==========================
# 🤖 ABA 0 - Processamento dos Documentos
# ==========================
with tab0:
    st.header("🤖 Processar Documentos com IA")

    if not st.session_state.csv_data:
        if st.button("Carregar dados..."):
            st.rerun()
        else:
            st.warning("Nenhum documento carregado! Por favor, faça o upload de um CSV na aba de 'Upload de base'.")
    elif not st.session_state.data:
        if st.button("Carregar dados..."):
            st.rerun()
        else:
            st.warning("Nenhuma pergunta cadastrada! Por favor, cadastre perguntas nas abas de 'Cadastro'.")
    else:
        model_options = [
            "databricks-meta-llama-3-3-70b-instruct",
            "databricks-meta-llama-3-1-405b-instruct",
            "databricks-mixtral-8x7b-instruct"
        ]
        selected_model = st.selectbox("Escolha o modelo de IA:", model_options, index=1)
        batch_size = st.slider("Escolha o tamanho do batch de perguntas:", 1, 50, 3)

        # Botão para iniciar o processamento
        if st.button("🔍 Executar Análise com IA"):
            results = []
            # 🔹 Inicializa a barra de progresso
            progress_bar = st.progress(0)
            total_docs = len(st.session_state.csv_data)
            processed_docs = 0
            for doc in st.session_state.csv_data:
                tipo_doc = doc.get("tipo_doc_rec", "")
                #tipo_doc = doc.get("tipo_doc", "")
                texto_total = doc.get("texto_total", "")
                if tipo_doc in st.session_state.data:
                    perguntas = st.session_state.data[tipo_doc].to_dict(orient="records")
                    dicionario_unico = {}
                    dicionario_unico["Número do Processo"] =  doc.get('numero_tj', '');
                    dicionario_unico["Tipo do documento"] =  doc.get("tipo_doc", "");
                    num_batches = math.ceil(len(perguntas) / batch_size)
                    for batch_index in range(num_batches):
                        batch_start = batch_index * batch_size
                        batch_end = min((batch_index + 1) * batch_size, len(perguntas))
                        batch = perguntas[batch_start:batch_end]
                        system_prompt = (
                            "Você é um assistente de IA especializado em extrair informações de documentos.\n"
                            "Para cada pergunta fornecida, responda no seguinte formato JSON:\n"
                            "{\n"
                            '  "<label_da_pergunta>": "<resposta_correspondente>"\n'
                            "}\n"
                            "Certifique-se de que cada resposta venha dentro de um único vetor de objetos JSON válido, sem escapes nem nada.\n"
                            "O vetor de retorno deve ser igual ao vetor de 'Perguntas' que te enviarei.\n"
                            "Agora vou te enviar o JSON com as perguntas e o Texto do Documento em JSON para que você me responda."
                        )
                        # 🔹 Criamos um único prompt estruturado para todas as perguntas do batch
                        prompt_dict = {
                            "Texto do Documento": texto_total,
                            "Perguntas": [
                                {
                                    "label_da_pergunta": pergunta['label'],
                                    "pergunta": pergunta['prompt']
                                }
                                for pergunta in batch
                            ]
                        }

                        # Convertendo o dicionário para uma string JSON formatada
                        prompt_structure = json.dumps(prompt_dict, ensure_ascii=False, indent=4)

                        # Enviar batch para o modelo de IA
                        chat_completion = client.chat.completions.create(
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": prompt_structure}
                            ],
                            model=selected_model,
                            max_tokens=4096
                        )

                        lista_de_dicts = extrair_json(chat_completion.choices[0].message.content)
                        if lista_de_dicts:
                            # Itera sobre a lista e adiciona elementos ao dicionário
                            for item in lista_de_dicts:
                                for dicionario in lista_de_dicts:
                                    dicionario_unico.update(dicionario)

                    results.append(dicionario_unico)

                # 🔹 Atualiza a barra de progresso após cada documento processado
                processed_docs += 1
                progress_bar.progress(processed_docs / total_docs)

            # 🔹 Esconde a barra de progresso ao finalizar
            progress_bar.empty()

            # 🔹 Converte para DataFrame
            st.session_state.results_df = pd.DataFrame(results)


    # 📊 Exibição da Tabela em Formato Wide
    if not st.session_state.results_df.empty:
        st.write("### 📊 Resultados da Análise")

        # Exibição da tabela no Streamlit
        #st.table(st.session_state.results_df)  # Para uma tabela estática
        st.dataframe(st.session_state.results_df)

# ==========================
# 📂 ABA 1 - Upload de CSV
# ==========================
with tab1:
    st.header("Upload de Base")

    uploaded_file = st.file_uploader("Escolha um arquivo CSV", type=["csv"])

    if uploaded_file:
        uploaded_file.seek(0)  # 🔹 Garante que o arquivo seja lido do início

        try:
            # 🔹 Lê o CSV com UTF-8 e detecta automaticamente o delimitador
            # df = pd.read_csv(uploaded_file, encoding="utf-8", sep=None, engine="python",
            #                  on_bad_lines="skip", skip_blank_lines=True)
            
            # substititu por esse código
            df = pd.read_csv(uploaded_file, encoding="utf-8", sep=','
                             , low_memory=False, on_bad_lines="skip", 
                             skip_blank_lines=True).dropna()

        except UnicodeDecodeError:
            st.error("Erro ao ler o arquivo. Certifique-se de que está em UTF-8.")

        except pd.errors.ParserError:
            st.error("Erro ao processar o CSV. Verifique se o delimitador está correto.")

        else:  # Executa apenas se não houver erro
            st.write("### Visualização dos Dados")
            st.dataframe(df)

            json_data = df.to_dict(orient="records")
            st.session_state.csv_data = json_data  # 🔹 Armazena os dados no session_state

            st.write("### JSON Gerado")
            st.json(json_data)

            json_str = json.dumps(json_data, indent=4, ensure_ascii=False)
            st.download_button(label="Baixar JSON", data=json_str, file_name="dados.json", mime="application/json")

# ==========================
# 📥 ABA 2 - Upload de JSON
# ==========================
#with tab2:
#    st.header("Cadastro via JSON")
#
#    uploaded_json = st.file_uploader("Selecione um arquivo de perguntas no formato JSON", type=["json"])
#
#    if uploaded_json:
#        st.session_state.uploaded_json = uploaded_json  # 🔹 Armazena o JSON apenas uma vez
#
#    # Verifica se o JSON já foi carregado antes de tentar processar
#    if st.session_state.uploaded_json:
#        try:
#            # 🔹 Verifica se o arquivo está vazio antes de tentar carregar
#            if st.session_state.uploaded_json is None or st.session_state.uploaded_json.size == 0:
#                st.error("Erro: O arquivo JSON está vazio ou inválido.")
#            else:
#                json_content = json.load(st.session_state.uploaded_json)
#
#                if isinstance(json_content, dict):
#                    for category, questions in json_content.items():
#                        if category not in st.session_state.data:
#                            st.session_state.data[category] = pd.DataFrame(questions)
#                        else:
#                            existing_df = st.session_state.data[category]
#                            new_df = pd.DataFrame(questions)
#                            merged_df = pd.concat([existing_df, new_df], ignore_index=True).drop_duplicates()
#                            st.session_state.data[category] = merged_df
#
#                    # 🔹 Adicionamos um botão para evitar recargas automáticas
#                    if st.button("Enviar Perguntas"):
#                        st.session_state.uploaded_json = None  # 🔹 Reseta o upload manualmente
#
#                else:
#                    st.error("Formato inválido! O JSON deve conter um dicionário de categorias.")
#
#        except json.JSONDecodeError:
#            st.error("Erro ao processar o JSON: O arquivo pode estar vazio ou mal formatado.")
#        except Exception as e:
#            st.error(f"Erro inesperado ao processar o JSON: {e}")
#
# ==========================
# 📋 ABA 3 - Cadastro
# ==========================
with tab3:
    st.header("Cadastro via Formulário")
    st.subheader("Gerenciar Categorias")

    # Adicionar nova categoria
    new_category = st.text_input("Adicionar nova categoria:", key="new_category_input")
    if st.button("➕ Adicionar Categoria", key="add_category_button") and new_category:
        if new_category not in st.session_state.data:
            st.session_state.data[new_category] = pd.DataFrame(columns=["label", "question", "prompt"])
            st.success(f"Categoria '{new_category}' adicionada com sucesso!")
        else:
            st.warning("Categoria já existe!")

    if st.session_state.data:
        selected_category = st.selectbox("Escolha uma categoria:", list(st.session_state.data.keys()), key="category_selectbox")

        # Botão para remover categoria
        if st.button("🗑️ Remover Categoria", key="remove_category_button"):
            del st.session_state.data[selected_category]
            st.success(f"Categoria '{selected_category}' removida com sucesso!")

        st.write("---")
        st.subheader(f"Adicionar Novo Registro em '{selected_category}'")

        with st.form("new_entry_form"):
            new_label = st.text_input("Label", key="new_label_input")
            new_question = st.text_input("Pergunta", key="new_question_input")
            new_prompt = st.text_area("Prompt", key="new_prompt_input")

            submitted = st.form_submit_button("Adicionar")

            if submitted and new_label and new_question and new_prompt:
                new_row = pd.DataFrame([{"label": new_label, "question": new_question, "prompt": new_prompt}])
                st.session_state.data[selected_category] = pd.concat([st.session_state.data[selected_category], new_row], ignore_index=True)
                st.success("Novo registro adicionado com sucesso!")

        st.write("---")

        # 📋 Exibição dos Dados + Remoção
        st.subheader(f"Dados Atuais em '{selected_category}'")

        df = st.session_state.data[selected_category]
        if not df.empty:
            cols = st.columns([3, 3, 3, 1])
            cols[0].write("**Label**")
            cols[1].write("**Pergunta**")
            cols[2].write("**Prompt**")
            cols[3].write("**Ação**")

            rows_to_remove = []
            for i, row in df.iterrows():
                cols = st.columns([3, 3, 3, 1])
                cols[0].write(row["label"])
                cols[1].write(row["question"])
                cols[2].write(row["prompt"])

                if cols[3].button("🗑️", key=f"remove_{selected_category}_{i}"):
                    rows_to_remove.append(i)

            if rows_to_remove:
                st.session_state.data[selected_category] = df.drop(rows_to_remove).reset_index(drop=True)
                st.success("Registro(s) removido(s) com sucesso!")
        else:
            st.info("Nenhum registro encontrado nesta categoria.")

        st.write("---")

        # 📥 Exportação dos Dados
        st.subheader("Exportar Dados")
        if st.button("📥 Baixar JSON", key="download_json_button"):
            json_data = {cat: df.to_dict(orient="records") for cat, df in st.session_state.data.items()}
            json_str = json.dumps(json_data, indent=4, ensure_ascii=False)
            st.download_button(label="Baixar JSON", data=json_str, file_name="dados.json", mime="application/json", key="download_json")

    st.write("---")
    # 📥 Importação dos Dados
    st.subheader("Importar Dados")
    uploaded_json = st.file_uploader("Selecione um arquivo de perguntas no formato JSON", type=["json"], key="json_file_uploader")
    if uploaded_json:
        try:
            json_content = json.load(uploaded_json)
            if isinstance(json_content, dict):
                for category, questions in json_content.items():
                    if category not in st.session_state.data:
                        st.session_state.data[category] = pd.DataFrame(questions)
                    else:
                        existing_df = st.session_state.data[category]
                        new_df = pd.DataFrame(questions)
                        merged_df = pd.concat([existing_df, new_df], ignore_index=True).drop_duplicates()
                        st.session_state.data[category] = merged_df
                st.success("Dados importados com sucesso!")
            else:
                st.error("Formato inválido! O JSON deve conter um dicionário de categorias.")
        except json.JSONDecodeError:
            st.error("Erro ao processar o JSON: O arquivo pode estar vazio ou mal formatado.")
        except Exception as e:
            st.error(f"Erro inesperado ao processar o JSON: {e}")

    if st.button("Atualizar"):
        st.rerun()
