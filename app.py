# ==============================================================================
# PLATAFORMA DE WORKFORCE MANAGEMENT (WFM) - VERSÃO 5.1 (CORREÇÃO DE LOCALE)
# Corrige o bug "unsupported locale setting" com tradução manual dos dias da semana.
# ==============================================================================
import streamlit as st
import pandas as pd
from pandas.errors import ParserError
import plotly.express as px
import math
from pyworkforce.queuing import ErlangC
import io
import csv
import chardet
from typing import Dict

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="WFM Pro - Otimizador de Escala",
    page_icon="🔮",
    layout="wide"
)

# ==============================================================================
# MÓDULO DE PROCESSAMENTO DE DADOS (LÓGICA DE BACKEND)
# ==============================================================================

@st.cache_data
def process_uploaded_file(uploaded_file, column_mapping):
    """Lê, valida e prepara o arquivo de histórico de chamadas usando o mapeamento de colunas."""
    file_bytes = uploaded_file.getvalue()
    try:
        sample = file_bytes[:8192]
        result = chardet.detect(sample)
        encoding = result['encoding'] or 'utf-8'
        text_sample = sample.decode(encoding, errors='ignore')
        sniffer = csv.Sniffer()
        delimiter = sniffer.sniff(text_sample).delimiter
    except Exception:
        encoding, delimiter = 'utf-8', ';'

    df = pd.read_csv(io.BytesIO(file_bytes), delimiter=delimiter, encoding=encoding, engine='python')
    
    df.rename(columns={v: k for k, v in column_mapping.items() if v is not None}, inplace=True)

    required_columns = {'data_hora_inicio', 'duracao_atendimento'}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Mapeamento incompleto. Faltam os campos obrigatórios: {missing}")

    df['data_hora_inicio'] = pd.to_datetime(df['data_hora_inicio'], format='%Y:%m:%d %H:%M:%S', errors='coerce')
    df['duracao_atendimento'] = pd.to_numeric(df['duracao_atendimento'], errors='coerce')
    
    linhas_antes = df.shape[0]
    df.dropna(subset=['data_hora_inicio', 'duracao_atendimento'], inplace=True)
    linhas_depois = df.shape[0]
    
    if df.empty:
        raise ValueError("Após a limpeza, nenhuma linha de dados válida restou. Verifique se o formato das datas ('AAAA:MM:DD HH:MM:SS') e durações (números) está correto em todo o arquivo.")

    # CORREÇÃO DE LOCALE: Pega o dia em inglês e traduz manualmente
    day_name_map_pt = {
        'Monday': 'Segunda-feira', 'Tuesday': 'Terça-feira', 'Wednesday': 'Quarta-feira',
        'Thursday': 'Quinta-feira', 'Friday': 'Sexta-feira', 'Saturday': 'Sábado', 'Sunday': 'Domingo'
    }
    df['Dia da Semana'] = df['data_hora_inicio'].dt.day_name().map(day_name_map_pt)
    df['Hora'] = df['data_hora_inicio'].dt.hour
    
    info_message = f"Arquivo processado. Linhas removidas: {linhas_antes - linhas_depois}. Linhas válidas: {linhas_depois}"
    return df, info_message

# ... (demais funções de cálculo permanecem as mesmas) ...
@st.cache_data
def calculate_demand(_df, growth_forecast):
    df = _df.copy()
    df['year'] = df['data_hora_inicio'].dt.year
    df['week'] = df['data_hora_inicio'].dt.isocalendar().week
    num_semanas = df[['year', 'week']].drop_duplicates().shape[0] or 1
    demand_df = df.groupby(['Dia da Semana', 'Hora']).agg(
        tma_medio=('duracao_atendimento', 'mean'),
        total_chamadas=('data_hora_inicio', 'count')).reset_index()
    demand_df['chamadas_por_hora'] = (demand_df['total_chamadas'] / num_semanas) * (1 + growth_forecast / 100)
    return demand_df

@st.cache_data
def _get_required_staff_binary_search(chamadas: float, tma: float, meta_sl: float, meta_tempo: int, max_positions: int = 100) -> int:
    low, high, best_n = 1, max_positions, max_positions
    erlang = ErlangC(transactions=chamadas, aht=tma, interval=3600, asa=meta_tempo)
    while low <= high:
        mid = (low + high) // 2
        if mid == 0: low = 1; continue
        if erlang.service_level(positions=mid) >= meta_sl:
            best_n, high = mid, mid - 1
        else:
            low = mid + 1
    return best_n

def calculate_required_staff(_demand_df, meta_sl, meta_tempo):
    required = [_get_required_staff_binary_search(float(row['chamadas_por_hora']), float(row['tma_medio'] or 0), meta_sl, meta_tempo) if float(row['chamadas_por_hora']) > 0 and float(row['tma_medio'] or 0) > 0 else 0 for _, row in _demand_df.iterrows()]
    return required

# ==============================================================================
# CAMADA DE APRESENTAÇÃO (UI - STREAMLIT)
# ==============================================================================

st.title("🔮 WFM Pro: Plataforma Analítica com Previsão")

# --- BARRA LATERAL ---
with st.sidebar:
    st.image("https://i.imgur.com/l33C2F3.png", width=200)
    st.header("1. Upload de Dados")
    uploaded_history = st.file_uploader("A. Histórico de Chamadas (CSV)", type="csv")
    
    column_mapping = {}
    if uploaded_history:
        header_df = pd.read_csv(uploaded_history, sep=None, engine='python', nrows=0)
        uploaded_history.seek(0)
        
        st.header("2. Mapeamento de Colunas")
        column_mapping['data_hora_inicio'] = st.selectbox("Data/Hora da Chamada", header_df.columns)
        column_mapping['duracao_atendimento'] = st.selectbox("Duração do Atendimento (em s)", header_df.columns)
        column_mapping['Fila/Cliente'] = st.selectbox("Fila/Cliente (Ex: Condomínio)", [None] + list(header_df.columns))
        column_mapping['Atendente'] = st.selectbox("Atendente", [None] + list(header_df.columns))
        column_mapping['Origem'] = st.selectbox("Origem da Chamada", [None] + list(header_df.columns))

    st.header("3. Metas e Parâmetros Globais")
    meta_sl = st.slider("Meta de Nível de Serviço (%)", 50, 100, 90) / 100.0
    meta_tempo = st.number_input("Tempo Alvo de Atendimento (s)", min_value=1, value=15)
    cost_per_hour = st.number_input("Custo por Hora de um Atendente (R$)", min_value=1.0, value=25.0)
    shrinkage = st.slider("Fator de Shrinkage (%)", 0, 50, 25) / 100.0
    growth_forecast = st.slider("Previsão de Crescimento de Volume (%)", -20, 100, 0)

# --- INICIALIZAÇÃO DO ESTADO ---
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

# --- LÓGICA PRINCIPAL ---
if uploaded_history:
    if st.button("📊 Processar e Analisar Dados", use_container_width=True, type="primary"):
        with st.spinner("Processando dados... Isso pode levar um momento."):
            try:
                df, info_message = process_uploaded_file(uploaded_history, column_mapping)
                st.session_state.processed_data = df
                st.success(info_message)
            except Exception as e:
                st.error(f"Ocorreu um erro: {e}")
                st.session_state.processed_data = None
else:
    st.info("Aguardando o upload do arquivo CSV para iniciar.")

# --- ABAS DE RESULTADOS (só aparecem se os dados foram processados) ---
if st.session_state.processed_data is not None:
    df = st.session_state.processed_data

    selected_queue = "Todos"
    if 'Fila/Cliente' in df.columns:
        queues = ["Todos"] + list(df['Fila/Cliente'].unique())
        selected_queue = st.selectbox("Filtrar Análise por Fila/Cliente:", queues)
    
    df_filtered = df[df['Fila/Cliente'] == selected_queue].copy() if selected_queue != "Todos" else df.copy()

    demand_df = calculate_demand(df_filtered, growth_forecast)
    demand_df['Atendentes Necessários'] = calculate_required_staff(demand_df, meta_sl, meta_tempo)
    demand_df['Escala Otimizada'] = demand_df['Atendentes Necessários'].apply(
        lambda x: 0 if x == 0 else math.ceil(x / (1 - shrinkage)))
    demand_df['Custo Otimizado (R$)'] = demand_df['Escala Otimizada'] * cost_per_hour

    tab1, tab2, tab3, tab4, tab5 = st.tabs([" dashboards Visão Geral", "🔍 Análise Detalhada", "💡 Otimização", "🚀 Performance", "🔮 Previsão e Simulação"])

    dias_ordem_pt = ['Segunda-feira', 'Terça-feira', 'Quarta-feira', 'Quinta-feira', 'Sexta-feira', 'Sábado', 'Domingo']

    with tab1:
        st.header(f"Dashboard de Performance: {selected_queue}")
        avg_calls = demand_df['chamadas_por_hora'].mean()
        avg_tma = df_filtered['duracao_atendimento'].mean()
        total_cost = demand_df['Custo Otimizado (R$)'].sum()
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Média de Chamadas/Hora", f"{avg_calls:.1f}")
        kpi2.metric("TMA Médio Geral", f"{avg_tma:.1f}s")
        kpi3.metric("Custo Semanal Otimizado", f"R$ {total_cost:,.2f}")
        hourly_demand_agg = demand_df.groupby('Hora', as_index=False)['chamadas_por_hora'].mean()
        fig_hourly = px.bar(hourly_demand_agg, x='Hora', y='chamadas_por_hora', title="Volume de Chamadas por Hora do Dia (Média Semanal)")
        st.plotly_chart(fig_hourly, use_container_width=True)

    with tab2:
        st.header(f"Análise Detalhada: {selected_queue}")
        selected_day = st.selectbox("Selecione um Dia:", dias_ordem_pt)
        day_df = demand_df[demand_df['Dia da Semana'] == selected_day]
        if not day_df.empty:
            fig_day = px.bar(day_df, x='Hora', y='chamadas_por_hora', title=f'Volume Horário - {selected_day}')
            st.plotly_chart(fig_day, use_container_width=True)
        else:
            st.warning(f"Não há dados para {selected_day}.")

    with tab3:
        st.header("Escala Otimizada Recomendada")
        st.dataframe(demand_df[['Dia da Semana', 'Hora', 'chamadas_por_hora', 'tma_medio', 'Escala Otimizada']].sort_values(by=['Dia da Semana', 'Hora']), hide_index=True)

    with tab4:
        st.header("Performance e Insights da Operação")
        if 'Atendente' in df.columns:
            st.subheader("Análise de Performance por Atendente")
            agent_perf = df_filtered.groupby('Atendente').agg(chamadas_atendidas=('data_hora_inicio', 'count'), tma_medio=('duracao_atendimento', 'mean')).sort_values(by='chamadas_atendidas', ascending=False).reset_index()
            fig_agent = px.bar(agent_perf, x='Atendente', y='chamadas_atendidas', color='tma_medio', title="Volume de Atendimentos vs. TMA Médio por Atendente")
            st.plotly_chart(fig_agent, use_container_width=True)
    
    with tab5:
        st.header("Previsão e Simulação de Cenários")
        with st.expander("🔬 Simulador Rápido (Independente)", expanded=True):
            st.write("Faça testes rápidos para qualquer cenário hipotético.")
            sim_col1, sim_col2, sim_col3 = st.columns(3)
            sim_chamadas = sim_col1.number_input("Volume de Chamadas/Hora", min_value=1.0, value=50.0, step=1.0)
            sim_tma = sim_col2.number_input("TMA (s)", min_value=1.0, value=180.0, step=5.0)
            sim_atendentes = sim_col3.number_input("Nº de Atendentes", min_value=1, value=4, step=1)
            if st.button("Simular Cenário Rápido"):
                erlang = ErlangC(transactions=sim_chamadas, aht=sim_tma, interval=3600, asa=meta_tempo)
                sl, ocupacao = erlang.service_level(positions=sim_atendentes), erlang.achieved_occupancy(positions=sim_atendentes)
                st.subheader("Resultado da Simulação Rápida")
                res_col1, res_col2 = st.columns(2)
                res_col1.metric("Nível de Serviço Previsto", f"{sl:.2%}")
                res_col2.metric("Ocupação dos Atendentes", f"{ocupacao:.2%}")
                if sl >= meta_sl: st.success("META ATINGIDA!")
                else: st.warning("ALERTA! A meta não será atingida.")

        with st.expander("📈 Simulador Baseado em Dados Históricos"):
            st.write("Selecione um horário específico da sua operação para simular com dados reais.")
            sim_col_data1, sim_col_data2 = st.columns(2)
            sel_dia = sim_col_data1.selectbox("Selecione o Dia da Semana", dias_ordem_pt, key="sim_dia")
            sel_hora = sim_col_data2.selectbox("Selecione a Hora", range(24), key="sim_hora")
            
            dados_reais = demand_df[(demand_df['Dia da Semana'] == sel_dia) & (demand_df['Hora'] == sel_hora)]
            
            if not dados_reais.empty:
                vol_real, tma_real = dados_reais['chamadas_por_hora'].iloc[0], dados_reais['tma_medio'].iloc[0]
                sim_col_data3, sim_col_data4 = st.columns(2)
                sim_col_data3.metric("Volume Real de Chamadas/Hora", f"{vol_real:.1f}")
                sim_col_data4.metric("TMA Médio Real (s)", f"{tma_real:.1f}")
                atendentes_sim_dados = st.slider("Nº de Atendentes para Simular", 1, 50, int(dados_reais['Escala Otimizada'].iloc[0] or 1))
                if st.button("Simular com Dados Reais"):
                    erlang_real = ErlangC(transactions=vol_real, aht=tma_real, interval=3600, asa=meta_tempo)
                    sl_real, ocupacao_real = erlang_real.service_level(positions=atendentes_sim_dados), erlang_real.achieved_occupancy(positions=atendentes_sim_dados)
                    st.subheader(f"Resultado da Simulação para {sel_dia} às {sel_hora}h")
                    res_col_real1, res_col_real2 = st.columns(2)
                    res_col_real1.metric("Nível de Serviço Previsto", f"{sl_real:.2%}")
                    res_col_real2.metric("Ocupação dos Atendentes", f"{ocupacao_real:.2%}")
                    if sl_real >= meta_sl: st.success("META ATINGIDA!")
                    else: st.warning("ALERTA! A meta não será atingida.")
            else:
                st.warning("Não há dados históricos para o dia/hora selecionado.")