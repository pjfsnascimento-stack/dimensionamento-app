# ==============================================================================
# PLATAFORMA DE WORKFORCE MANAGEMENT (WFM) - VERSÃO 3.0 (PROFISSIONAL)
# Incorpora análise técnica, correções de bugs e otimizações de performance.
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
    page_icon="🚀",
    layout="wide"
)

# ==============================================================================
# MÓDULO DE PROCESSAMENTO DE DADOS (LÓGICA DE BACKEND)
# Funções isoladas da UI, lançando exceções para serem tratadas na camada de apresentação.
# ==============================================================================

def detect_csv_params(file_bytes: bytes) -> Dict[str, any]:
    """Detecta encoding e delimitador de um arquivo CSV em bytes."""
    try:
        sample = file_bytes[:8192]
        result = chardet.detect(sample)
        encoding = result['encoding'] or 'utf-8'
        text_sample = sample.decode(encoding, errors='ignore')
        sniffer = csv.Sniffer()
        delimiter = sniffer.sniff(text_sample).delimiter
    except Exception:
        # Fallback para os padrões mais comuns
        encoding = 'utf-8'
        delimiter = ','
    return {"encoding": encoding, "delimiter": delimiter}

@st.cache_data
def process_uploaded_file(uploaded_file):
    """Lê, valida e prepara o arquivo de histórico de chamadas."""
    file_bytes = uploaded_file.getvalue()
    csv_params = detect_csv_params(file_bytes)
    
    df = pd.read_csv(io.BytesIO(file_bytes), delimiter=csv_params["delimiter"], encoding=csv_params["encoding"], engine='python')
    
    required_columns = {'data_hora_inicio', 'duracao_atendimento'}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Arquivo CSV inválido. Faltam as colunas obrigatórias: {missing}")

    df['data_hora_inicio'] = pd.to_datetime(df['data_hora_inicio'], errors='coerce')
    df['duracao_atendimento'] = pd.to_numeric(df['duracao_atendimento'], errors='coerce')
    
    linhas_antes = df.shape[0]
    df.dropna(subset=['data_hora_inicio', 'duracao_atendimento'], inplace=True)
    linhas_depois = df.shape[0]
    
    if df.empty:
        raise ValueError("Após a limpeza, nenhuma linha de dados válida restou. Verifique o conteúdo do arquivo.")

    df['Dia da Semana'] = df['data_hora_inicio'].dt.day_name(locale='pt_BR.utf8')
    df['Hora'] = df['data_hora_inicio'].dt.hour
    
    info_message = f"Arquivo processado. Linhas removidas: {linhas_antes - linhas_depois}. Linhas válidas: {linhas_depois}"
    return df, info_message

@st.cache_data
def calculate_demand(_df, growth_forecast):
    """Calcula a demanda de chamadas e TMA por hora, de forma estatisticamente correta."""
    df = _df.copy()
    df['year'] = df['data_hora_inicio'].dt.year
    df['week'] = df['data_hora_inicio'].dt.isocalendar().week
    num_semanas = df[['year', 'week']].drop_duplicates().shape[0] or 1

    demand_df = df.groupby(['Dia da Semana', 'Hora']).agg(
        tma_medio=('duracao_atendimento', 'mean'),
        total_chamadas=('data_hora_inicio', 'count')
    ).reset_index()
    demand_df['chamadas_por_hora'] = (demand_df['total_chamadas'] / num_semanas) * (1 + growth_forecast / 100)
    
    return demand_df

@st.cache_data
def _get_required_staff_binary_search(chamadas: float, tma: float, meta_sl: float, meta_tempo: int, max_positions: int = 100) -> int:
    """Usa busca binária para encontrar o número ótimo de atendentes, com cache."""
    low, high = 1, max_positions
    best_n = max_positions
    
    erlang = ErlangC(transactions=chamadas, aht=tma, interval=3600, asa=meta_tempo)

    while low <= high:
        mid = (low + high) // 2
        if mid == 0: # Evitar posições = 0
            low = 1
            continue
        
        sl = erlang.service_level(positions=mid)
        if sl >= meta_sl:
            best_n = mid
            high = mid - 1
        else:
            low = mid + 1
            
    return best_n

def calculate_required_staff(_demand_df, meta_sl, meta_tempo):
    """Aplica o cálculo otimizado para cada intervalo de tempo."""
    required = []
    for _, row in _demand_df.iterrows():
        chamadas, tma = float(row['chamadas_por_hora']), float(row['tma_medio'] or 0)
        if chamadas <= 0 or tma <= 0:
            required.append(0)
            continue
        n = _get_required_staff_binary_search(chamadas, tma, meta_sl, meta_tempo)
        required.append(n)
    return required

# ==============================================================================
# CAMADA DE APRESENTAÇÃO (UI - STREAMLIT)
# ==============================================================================

st.title("🚀 WFM Pro: Otimizador de Escala Inteligente")
st.markdown("Transforme dados em eficiência. Faça o upload do seu histórico de chamadas e da sua escala atual para receber uma análise completa e recomendações de otimização.")

with st.sidebar:
    st.image("https://i.imgur.com/l33C2F3.png", width=200)
    st.header("1. Upload de Dados")
    uploaded_history = st.file_uploader("A. Histórico de Chamadas (CSV)", type="csv", help="Deve conter as colunas 'data_hora_inicio' e 'duracao_atendimento'")
    uploaded_current_scale = st.file_uploader("B. Escala Atual (Opcional, CSV)", type="csv", help="Deve conter as colunas 'Dia da Semana', 'Hora', 'Atendentes'")
    
    st.header("2. Metas Operacionais")
    meta_sl = st.slider("Meta de Nível de Serviço (%)", 50, 100, 90) / 100.0
    meta_tempo = st.number_input("Tempo Alvo de Atendimento (s)", min_value=1, value=15)
    
    st.header("3. Parâmetros de Custo e Previsão")
    cost_per_hour = st.number_input("Custo por Hora de um Atendente (R$)", min_value=1.0, value=25.0, step=0.5)
    shrinkage = st.slider("Fator de Shrinkage (%)", 0, 50, 25) / 100.0
    growth_forecast = st.slider("Previsão de Crescimento de Volume (%)", -20, 100, 0, help="Aplica um fator de crescimento sobre o volume histórico.")

if uploaded_history is not None:
    try:
        df, info_message = process_uploaded_file(uploaded_history)
        st.success(info_message)
        
        demand_df = calculate_demand(df, growth_forecast)
        demand_df['Atendentes Necessários'] = calculate_required_staff(demand_df, meta_sl, meta_tempo)
        demand_df['Escala Otimizada'] = demand_df['Atendentes Necessários'].apply(
            lambda x: 0 if x == 0 else math.ceil(x / (1 - shrinkage))
        )
        demand_df['Custo Otimizado (R$)'] = demand_df['Escala Otimizada'] * cost_per_hour

        tab1, tab2, tab3, tab4 = st.tabs([" dashboards Visão Geral", "🔍 Análise Detalhada", "💡 Otimização de Escala", "🔧 Simulador de Cenários"])

        with tab1:
            st.header("Dashboard de Performance da Operação")
            
            # KPI Corrigido
            avg_calls_per_hour = demand_df['chamadas_por_hora'].mean()
            avg_tma = df['duracao_atendimento'].mean()
            total_optimized_cost = demand_df['Custo Otimizado (R$)'].sum()

            kpi1, kpi2, kpi3 = st.columns(3)
            kpi1.metric("Média de Chamadas/Hora", f"{avg_calls_per_hour:.1f}")
            kpi2.metric("TMA Médio Geral", f"{avg_tma:.1f}s")
            kpi3.metric("Custo Semanal Otimizado", f"R$ {total_optimized_cost:,.2f}")

            st.subheader("Volume de Chamadas por Hora do Dia (Média Semanal)")
            # Plot corrigido
            hourly_demand_agg = demand_df.groupby('Hora', as_index=False)['chamadas_por_hora'].mean()
            fig_hourly = px.bar(hourly_demand_agg, x='Hora', y='chamadas_por_hora', labels={'Hora': 'Hora do Dia', 'chamadas_por_hora': 'Volume Médio de Chamadas'})
            st.plotly_chart(fig_hourly, use_container_width=True)

        with tab2:
            st.header("Análise Detalhada do Volume e TMA")
            dias_ordem_pt = ['Segunda-feira', 'Terça-feira', 'Quarta-feira', 'Quinta-feira', 'Sexta-feira', 'Sábado', 'Domingo']
            selected_day = st.selectbox("Selecione um Dia da Semana para detalhar:", dias_ordem_pt)
            
            day_df = demand_df[demand_df['Dia da Semana'] == selected_day]
            if not day_df.empty:
                fig_day_demand = px.bar(day_df, x='Hora', y='chamadas_por_hora', title=f'Volume de Chamadas por Hora - {selected_day}', labels={'Hora': 'Hora do Dia', 'chamadas_por_hora': 'Volume de Chamadas'})
                st.plotly_chart(fig_day_demand, use_container_width=True)
                
                fig_day_tma = px.line(day_df, x='Hora', y='tma_medio', title=f'Variação do TMA por Hora - {selected_day}', labels={'Hora': 'Hora do Dia', 'tma_medio': 'TMA Médio (s)'}, markers=True)
                st.plotly_chart(fig_day_tma, use_container_width=True)
            else:
                st.warning(f"Não há dados para {selected_day} no histórico fornecido.")

        with tab3:
            st.header("Comparativo de Escalas: Atual vs. Otimizada")
            
            if uploaded_current_scale is not None:
                current_scale_df = pd.read_csv(uploaded_current_scale, sep=None, engine='python')
                current_scale_df.rename(columns={'Atendentes': 'Escala Atual'}, inplace=True)
                
                comparison_df = pd.merge(demand_df, current_scale_df, on=['Dia da Semana', 'Hora'], how='left').fillna(0)
                comparison_df['Custo Atual (R$)'] = comparison_df['Escala Atual'] * cost_per_hour
                comparison_df['Diferença (Headcount)'] = comparison_df['Escala Otimizada'] - comparison_df['Escala Atual']
                
                total_current_cost = comparison_df['Custo Atual (R$)'].sum()
                savings = total_current_cost - total_optimized_cost
                
                st.subheader("Análise Financeira e de Headcount")
                comp_kpi1, comp_kpi2, comp_kpi3 = st.columns(3)
                comp_kpi1.metric("Custo Semanal Atual", f"R$ {total_current_cost:,.2f}")
                comp_kpi2.metric("Custo Semanal Otimizado", f"R$ {total_optimized_cost:,.2f}")
                comp_kpi3.metric("Potencial de Economia Semanal", f"R$ {savings:,.2f}", delta=f"{savings:,.2f}")

                st.dataframe(comparison_df[['Dia da Semana', 'Hora', 'Escala Atual', 'Escala Otimizada', 'Diferença (Headcount)', 'Custo Atual (R$)', 'Custo Otimizado (R$)']].sort_values(by=['Dia da Semana', 'Hora']), hide_index=True)
                
            else:
                st.info("Faça o upload da sua escala atual na barra lateral para ver a análise comparativa.")
                st.subheader("Escala Otimizada Recomendada")
                st.dataframe(demand_df[['Dia da Semana', 'Hora', 'chamadas_por_hora', 'tma_medio', 'Escala Otimizada']].sort_values(by=['Dia da Semana', 'Hora']), hide_index=True)

        with tab4:
            st.header("Simulador de Cenários Pontuais")
            sim_col1, sim_col2, sim_col3 = st.columns(3)
            sim_chamadas = sim_col1.number_input("Volume de Chamadas/Hora", min_value=1.0, value=50.0, step=1.0, key="sim_calls")
            sim_tma = sim_col2.number_input("TMA (s)", min_value=1.0, value=180.0, step=5.0, key="sim_aht")
            sim_atendentes = sim_col3.number_input("Nº de Atendentes", min_value=1, value=4, step=1, key="sim_agents")

            if st.button("Simular Cenário"):
                erlang_sim = ErlangC(transactions=sim_chamadas, aht=sim_tma, interval=3600, asa=meta_tempo)
                sl_previsto = erlang_sim.service_level(positions=sim_atendentes)
                ocupacao = erlang_sim.achieved_occupancy(positions=sim_atendentes)
                st.subheader("Resultado da Simulação")
                res_col1, res_col2 = st.columns(2)
                res_col1.metric("Nível de Serviço Previsto", f"{sl_previsto:.2%}")
                res_col2.metric("Ocupação dos Atendentes", f"{ocupacao:.2%}")
                if sl_previsto >= meta_sl: st.success("META ATINGIDA!")
                else: st.warning("ALERTA! A meta não será atingida.")
                
    except Exception as e:
        st.error(f"Ocorreu um erro durante o processamento: {e}")
        st.warning("Verifique se o seu arquivo CSV está bem formatado e contém as colunas necessárias.")

else:
    st.info("Aguardando o upload do arquivo de histórico de chamadas para iniciar a análise.")