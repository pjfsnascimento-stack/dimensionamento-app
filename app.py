# ==============================================================================
# APLICAÇÃO DE DIMENSIONAMENTO DE EQUIPE COM STREAMLIT - VERSÃO FINAL
# Para executar localmente: no terminal, navegue até a pasta deste arquivo e digite:
# streamlit run app.py
# ==============================================================================
import streamlit as st
import pandas as pd
from pandas.errors import ParserError # <-- CORREÇÃO APLICADA AQUI
import matplotlib.pyplot as plt
import seaborn as sns
import math
from pyworkforce.queuing import ErlangC

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Dimensionamento de Central",
    page_icon="📞",
    layout="wide"
)

# --- FUNÇÕES DE LÓGICA (Baseado nos módulos do Colab) ---

def clean_and_prepare_data(df):
    """Pega o DataFrame bruto e realiza toda a limpeza e preparação."""
    st.write("Iniciando limpeza e preparação dos dados...")
    
    # Tenta adivinhar o formato da data, com fallback para o formato especial
    try:
        df['data_hora_inicio'] = pd.to_datetime(df['data_hora_inicio'], format='%Y:%m:%d %H:%M:%S', errors='coerce')
    except (TypeError, ValueError):
        df['data_hora_inicio'] = pd.to_datetime(df['data_hora_inicio'], errors='coerce')
        
    df['duracao_atendimento'] = pd.to_numeric(df['duracao_atendimento'], errors='coerce')
    
    linhas_antes = df.shape[0]
    df.dropna(subset=['data_hora_inicio', 'duracao_atendimento'], inplace=True)
    linhas_depois = df.shape[0]
    
    st.info(f"Limpeza de dados concluída. Linhas removidas por dados inválidos: {linhas_antes - linhas_depois}. Linhas válidas restantes: {linhas_depois}")

    if df.empty:
        st.error("Após a limpeza, nenhuma linha de dados válida restou. Verifique o formato das colunas 'data_hora_inicio' e 'duracao_atendimento' no seu CSV.")
        return None

    df['dia_da_semana'] = df['data_hora_inicio'].dt.day_name()
    df['hora'] = df['data_hora_inicio'].dt.hour
    
    return df

def calculate_staffing(df, meta_sl, meta_tempo, shrinkage):
    """Calcula a necessidade de atendentes com base nos dados e metas."""
    st.write("Calculando o dimensionamento...")

    dias_no_historico = (df['data_hora_inicio'].max() - df['data_hora_inicio'].min()).days
    num_semanas = dias_no_historico / 7 if dias_no_historico > 0 else 1
    if num_semanas < 1: num_semanas = 1

    analise_horaria = df.groupby(['dia_da_semana', 'hora']).agg(
        tma_medio=('duracao_atendimento', 'mean'),
        total_chamadas=('data_hora_inicio', 'count')
    ).reset_index()
    analise_horaria['chamadas_por_hora'] = analise_horaria['total_chamadas'] / num_semanas
    
    escala_recomendada = []
    for _, row in analise_horaria.iterrows():
        chamadas_hora, tma = row['chamadas_por_hora'], row['tma_medio']
        if chamadas_hora > 0 and tma > 0:
            for n in range(1, 100):
                erlang = ErlangC(transactions=chamadas_hora, aht=tma, interval=3600, asa=meta_tempo)
                if erlang.service_level(positions=n) >= meta_sl:
                    atendentes_necessarios = n
                    break
            else:
                atendentes_necessarios = 99
        else:
            atendentes_necessarios = 0
            
        atendentes_na_escala = 0 if atendentes_necessarios == 0 else math.ceil(atendentes_necessarios / (1 - shrinkage))
        
        escala_recomendada.append({
            'Dia da Semana': row['dia_da_semana'], 'Hora': row['hora'],
            'Atendentes Necessários': atendentes_necessarios,
            'Atendentes na Escala (com Shrinkage)': atendentes_na_escala
        })
        
    return pd.DataFrame(escala_recomendada)

# --- INTERFACE DA APLICAÇÃO ---

st.title("👨‍💻 Plataforma de Dimensionamento para Central de Atendimento")
st.write("Faça o upload do seu histórico de chamadas em formato CSV para gerar a escala e fazer simulações.")

# --- BARRA LATERAL PARA INPUTS ---
with st.sidebar:
    st.header("1. Configurações")
    uploaded_file = st.file_uploader("Selecione o arquivo CSV", type="csv")
    
    st.header("2. Metas e Parâmetros")
    meta_sl = st.slider("Meta de Nível de Serviço (%)", 50, 100, 90) / 100.0
    meta_tempo = st.number_input("Tempo Alvo de Atendimento (segundos)", min_value=1, value=15)
    shrinkage = st.slider("Fator de Shrinkage (%)", 0, 50, 25) / 100.0
    
    st.info("""
    **Nível de Serviço:** % de chamadas atendidas dentro do tempo alvo.
    **Shrinkage:** % de tempo em que o atendente não está disponível (pausas, etc.).
    """)

# --- LÓGICA PRINCIPAL E EXIBIÇÃO DE RESULTADOS ---
if uploaded_file is not None:
    try:
        # Tenta detectar o separador automaticamente
        df_bruto = pd.read_csv(uploaded_file, sep=None, engine='python')
        
        st.header("Análise e Dimensionamento")
        df_preparado = clean_and_prepare_data(df_bruto)
        
        if df_preparado is not None:
            df_escala = calculate_staffing(df_preparado, meta_sl, meta_tempo, shrinkage)
            
            st.success("Dimensionamento concluído com sucesso!")
            
            # Mapear dias da semana para ordenar
            dias_ordem = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            df_escala['Dia da Semana'] = pd.Categorical(df_escala['Dia da Semana'], categories=dias_ordem, ordered=True)
            df_escala = df_escala.sort_values(['Dia da Semana', 'Hora'])

            st.dataframe(df_escala, hide_index=True, use_container_width=True)

            # Visualizações
            st.subheader("Visualização da Escala (com Shrinkage)")
            escala_pivot = df_escala.pivot_table(
                index='Hora', 
                columns='Dia da Semana', 
                values='Atendentes na Escala (com Shrinkage)'
            )
            
            # Garantir a ordem correta na visualização
            colunas_existentes_ordenadas = [dia for dia in dias_ordem if dia in escala_pivot.columns]
            escala_pivot = escala_pivot[colunas_existentes_ordenadas]

            fig, ax = plt.subplots(figsize=(15, 10))
            sns.heatmap(escala_pivot.fillna(0).astype(int), annot=True, fmt='d', cmap='YlGnBu', linewidths=.5, ax=ax)
            ax.set_title('Mapa de Calor da Escala de Atendentes Necessários')
            st.pyplot(fig)
            
            # Botão para download
            @st.cache_data
            def convert_df_to_csv(df):
                return df.to_csv(index=False).encode('utf-8')

            csv = convert_df_to_csv(df_escala)
            st.download_button(
                label="Baixar escala completa em CSV",
                data=csv,
                file_name='escala_recomendada.csv',
                mime='text/csv',
            )

    except (ParserError, UnicodeDecodeError, ValueError) as e:
        st.error(f"Ocorreu um erro ao ler o arquivo CSV. Verifique se o separador é ',' ou ';'. Detalhe do erro: {e}")
    except Exception as e:
        st.error(f"Ocorreu um erro inesperado ao processar o arquivo: {e}")

# --- SEÇÃO DE SIMULAÇÃO ---
st.header("🔧 Ferramenta de Simulação Rápida")
st.write("Faça testes 'E se...?' para prever o Nível de Serviço em diferentes cenários.")

col1, col2, col3 = st.columns(3)
with col1:
    sim_chamadas = st.number_input("Volume de Chamadas/Hora", min_value=1.0, value=50.0, step=1.0)
with col2:
    sim_tma = st.number_input("TMA (segundos)", min_value=1.0, value=180.0, step=5.0)
with col3:
    sim_atendentes = st.number_input("Nº de Atendentes", min_value=1, value=4, step=1)

if st.button("Simular Cenário"):
    erlang = ErlangC(transactions=sim_chamadas, aht=sim_tma, interval=3600, asa=meta_tempo)
    sl_previsto = erlang.service_level(positions=sim_atendentes)
    ocupacao = erlang.achieved_occupancy(positions=sim_atendentes)
    
    st.subheader("Resultado da Simulação")
    col_res1, col_res2 = st.columns(2)
    with col_res1:
        st.metric("Nível de Serviço Previsto", f"{sl_previsto:.2%}")
    with col_res2:
        st.metric("Ocupação dos Atendentes", f"{ocupacao:.2%}")
    
    if sl_previsto >= meta_sl:
        st.success("META ATINGIDA! A equipe está bem dimensionada para este cenário.")
    else:
        st.warning("ALERTA! A meta de serviço não será atingida neste cenário.")