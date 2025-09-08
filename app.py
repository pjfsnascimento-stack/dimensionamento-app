# ==============================================================================
# APLICAÇÃO DE DIMENSIONAMENTO - VERSÃO DE DEPURAÇÃO
# ==============================================================================
import streamlit as st
import pandas as pd
from pandas.errors import ParserError
import matplotlib.pyplot as plt
import seaborn as sns
import math
from pyworkforce.queuing import ErlangC
import io

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Debug - Dimensionamento",
    page_icon="🐞",
    layout="wide"
)

# --- FUNÇÕES (sem alterações) ---
# ... (As funções clean_and_prepare_data e calculate_staffing são as mesmas da versão anterior) ...
def clean_and_prepare_data(df):
    try:
        df['data_hora_inicio_convertida'] = pd.to_datetime(df['data_hora_inicio'], format='%Y:%m:%d %H:%M:%S', errors='coerce')
    except (TypeError, ValueError):
        df['data_hora_inicio_convertida'] = pd.to_datetime(df['data_hora_inicio'], errors='coerce')
    df['duracao_atendimento_convertida'] = pd.to_numeric(df['duracao_atendimento'], errors='coerce')
    return df

# --- TÍTULO E INTRODUÇÃO ---
st.title("🐞 Modo de Depuração - Análise de CSV")
st.write("Esta é uma versão especial da aplicação para diagnosticar problemas com o formato do arquivo CSV.")

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("Upload de Arquivo")
    uploaded_file = st.file_uploader("Selecione o arquivo CSV para análise", type="csv")
    debug_mode = st.checkbox("Ativar Modo de Depuração", value=True)

# --- LÓGICA PRINCIPAL ---
if uploaded_file is not None and debug_mode:
    try:
        st.header("Resultados da Depuração")
        
        # 1. Leitura do arquivo
        df_raw = pd.read_csv(uploaded_file, sep=None, engine='python')
        st.subheader("1. Amostra dos Dados Brutos (Como foram lidos)")
        st.write("Abaixo estão as 10 primeiras linhas do seu arquivo, exatamente como o Python as leu. Verifique se as colunas estão separadas corretamente.")
        st.dataframe(df_raw.head(10))

        # 2. Informações das Colunas
        st.subheader("2. Tipos de Dados das Colunas (Antes da Conversão)")
        st.write("Isso nos mostra como o Pandas classificou cada coluna. `object` geralmente significa texto.")
        buffer = io.StringIO()
        df_raw.info(buf=buffer)
        st.text(buffer.getvalue())

        # 3. Tentativa de Conversão
        st.subheader("3. Análise da Conversão (A parte mais importante)")
        st.write("Aqui, tentamos converter as colunas `data_hora_inicio` e `duracao_atendimento`. `NaT` significa que a data falhou na conversão. `NaN` significa que o número falhou.")
        
        df_debug = clean_and_prepare_data(df_raw.copy())
        
        # Seleciona apenas as colunas relevantes para a depuração
        colunas_debug = [
            'data_hora_inicio', 
            'data_hora_inicio_convertida', 
            'duracao_atendimento', 
            'duracao_atendimento_convertida'
        ]
        st.dataframe(df_debug[colunas_debug].head(20))

        # 4. Diagnóstico Final
        st.subheader("4. Diagnóstico Final")
        datas_validas = df_debug['data_hora_inicio_convertida'].notna().sum()
        duracoes_validas = df_debug['duracao_atendimento_convertida'].notna().sum()
        total_linhas = len(df_debug)

        st.write(f" - **Total de Linhas no Arquivo:** {total_linhas}")
        st.write(f" - **Linhas com 'data_hora_inicio' VÁLIDA:** {datas_validas}")
        st.write(f" - **Linhas com 'duracao_atendimento' VÁLIDA:** {duracoes_validas}")

        if datas_validas == 0 or duracoes_validas == 0:
            st.error("Diagnóstico: Pelo menos uma das colunas chave ('data_hora_inicio' ou 'duracao_atendimento') falhou na conversão para TODAS as linhas. Isso causa o erro 'nenhuma linha válida restou'.")
            if datas_validas == 0:
                st.warning("A coluna `data_hora_inicio` parece ser a principal culpada. Verifique se o formato `AAAA:MM:DD HH:MM:SS` está consistente em TODO o arquivo, sem textos ou caracteres inesperados.")
        else:
            st.success("Diagnóstico: A conversão parece ter funcionado para pelo menos algumas linhas. O problema pode ser outro.")

    except Exception as e:
        st.error(f"Ocorreu um erro durante a depuração: {e}")
        
elif uploaded_file is not None and not debug_mode:
    st.info("Por favor, marque a caixa 'Ativar Modo de Depuração' na barra lateral para analisar o arquivo.")

else:
    st.info("Aguardando o upload do arquivo CSV para iniciar a depuração.")