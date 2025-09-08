# ==============================================================================
# PLATAFORMA DE WORKFORCE MANAGEMENT (WFM) - VERSÃO 2.0
# Por: Gemini
# ==============================================================================
import streamlit as st
import pandas as pd
from pandas.errors import ParserError
import plotly.express as px
import math
from pyworkforce.queuing import ErlangC
import io

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="WFM Pro - Otimizador de Escala",
    page_icon="📊",
    layout="wide"
)

# --- FUNÇÕES DE LÓGICA (O motor da aplicação) ---

@st.cache_data
def process_uploaded_file(uploaded_file):
    """Lê e prepara o arquivo de histórico de chamadas."""
    try:
        df = pd.read_csv(uploaded_file, sep=None, engine='python')
    except (ParserError, UnicodeDecodeError, ValueError) as e:
        st.error(f"Erro ao ler o arquivo CSV. Verifique o separador (deve ser ',' ou ';'). Detalhe: {e}")
        return None, None

    try:
        df['data_hora_inicio'] = pd.to_datetime(df['data_hora_inicio'], format='%Y:%m:%d %H:%M:%S', errors='coerce')
    except (TypeError, ValueError):
        df['data_hora_inicio'] = pd.to_datetime(df['data_hora_inicio'], errors='coerce')
        
    df['duracao_atendimento'] = pd.to_numeric(df['duracao_atendimento'], errors='coerce')
    
    linhas_antes = df.shape[0]
    df.dropna(subset=['data_hora_inicio', 'duracao_atendimento'], inplace=True)
    linhas_depois = df.shape[0]

    info_message = f"Arquivo processado. Linhas removidas por dados inválidos: {linhas_antes - linhas_depois}. Linhas válidas para análise: {linhas_depois}"
    
    if df.empty:
        st.error("Após a limpeza, nenhuma linha de dados válida restou. Verifique o formato das colunas 'data_hora_inicio' e 'duracao_atendimento'.")
        return None, None

    df['Dia da Semana'] = df['data_hora_inicio'].dt.day_name()
    df['Hora'] = df['data_hora_inicio'].dt.hour
    
    return df, info_message

@st.cache_data
def calculate_demand(df, growth_forecast):
    """Calcula a demanda de chamadas e TMA por hora, aplicando a previsão de crescimento."""
    dias_no_historico = (df['data_hora_inicio'].max() - df['data_hora_inicio'].min()).days
    num_semanas = dias_no_historico / 7 if dias_no_historico > 0 else 1
    if num_semanas < 1: num_semanas = 1

    demand_df = df.groupby(['Dia da Semana', 'Hora']).agg(
        tma_medio=('duracao_atendimento', 'mean'),
        total_chamadas=('data_hora_inicio', 'count')
    ).reset_index()
    demand_df['chamadas_por_hora'] = (demand_df['total_chamadas'] / num_semanas) * (1 + growth_forecast / 100)
    
    return demand_df

def calculate_required_staff(_demand_df, meta_sl, meta_tempo):
    """Calcula o número de atendentes necessários (sem shrinkage) para cada intervalo."""
    required_staff = []
    for _, row in _demand_df.iterrows():
        chamadas, tma = row['chamadas_por_hora'], row['tma_medio']
        if chamadas > 0 and tma > 0:
            for n in range(1, 100):
                erlang = ErlangC(transactions=chamadas, aht=tma, interval=3600, asa=meta_tempo)
                if erlang.service_level(positions=n) >= meta_sl:
                    required_staff.append(n)
                    break
            else:
                required_staff.append(99)
        else:
            required_staff.append(0)
    return required_staff

# --- INTERFACE DA APLICAÇÃO ---

st.title("📊 WFM Pro: Otimizador de Escala Inteligente")
st.markdown("Transforme dados em eficiência. Faça o upload do seu histórico de chamadas e da sua escala atual para receber uma análise completa e recomendações de otimização.")

# --- BARRA LATERAL PARA INPUTS ---
with st.sidebar:
    st.image("https://i.imgur.com/l33C2F3.png", width=200) # Exemplo de logo
    st.header("1. Upload de Dados")
    uploaded_history = st.file_uploader("A. Histórico de Chamadas (CSV)", type="csv")
    uploaded_current_scale = st.file_uploader("B. Escala Atual (Opcional, CSV)", type="csv")
    
    st.header("2. Metas Operacionais")
    meta_sl = st.slider("Meta de Nível de Serviço (%)", 50, 100, 90) / 100.0
    meta_tempo = st.number_input("Tempo Alvo de Atendimento (s)", min_value=1, value=15)
    
    st.header("3. Parâmetros de Custo e Previsão")
    cost_per_hour = st.number_input("Custo por Hora de um Atendente (R$)", min_value=1.0, value=25.0, step=0.5)
    shrinkage = st.slider("Fator de Shrinkage (%)", 0, 50, 25) / 100.0
    growth_forecast = st.slider("Previsão de Crescimento de Volume (%)", -20, 100, 0)
    
    st.info("A escala atual deve ter 3 colunas: 'Dia da Semana', 'Hora', 'Atendentes'.")

# --- LÓGICA PRINCIPAL ---
if uploaded_history is not None:
    df, info_message = process_uploaded_file(uploaded_history)
    
    if df is not None:
        st.info(info_message)
        
        demand_df = calculate_demand(df, growth_forecast)
        demand_df['Atendentes Necessários'] = calculate_required_staff(demand_df, meta_sl, meta_tempo)
        demand_df['Escala Otimizada'] = demand_df['Atendentes Necessários'].apply(
            lambda x: 0 if x == 0 else math.ceil(x / (1 - shrinkage))
        )
        demand_df['Custo Otimizado (R$)'] = demand_df['Escala Otimizada'] * cost_per_hour

        # --- ABAS DA INTERFACE ---
        tab1, tab2, tab3, tab4 = st.tabs([" dashboards Visão Geral", "🔍 Análise Detalhada", "💡 Otimização de Escala", "🔧 Simulador de Cenários"])

        # --- ABA 1: VISÃO GERAL ---
        with tab1:
            st.header("Dashboard de Performance da Operação")
            
            total_calls = demand_df['chamadas_por_hora'].sum() * 24 * 7 / (demand_df.shape[0])
            avg_tma = df['duracao_atendimento'].mean()
            total_optimized_cost = demand_df['Custo Otimizado (R$)'].sum()

            kpi1, kpi2, kpi3 = st.columns(3)
            kpi1.metric("Média de Chamadas/Hora", f"{total_calls:.1f}")
            kpi2.metric("TMA Médio Geral", f"{avg_tma:.1f}s")
            kpi3.metric("Custo Semanal Otimizado", f"R$ {total_optimized_cost:,.2f}")

            st.subheader("Volume de Chamadas por Hora do Dia (Média Semanal)")
            hourly_demand = demand_df.groupby('Hora')['chamadas_por_hora'].mean()
            fig_hourly = px.bar(hourly_demand, x=hourly_demand.index, y='chamadas_por_hora', labels={'Hora': 'Hora do Dia', 'chamadas_por_hora': 'Volume Médio de Chamadas'})
            st.plotly_chart(fig_hourly, use_container_width=True)

        # --- ABA 2: ANÁLISE DETALHADA ---
        with tab2:
            st.header("Análise Detalhada do Volume e TMA")
            selected_day = st.selectbox("Selecione um Dia da Semana para detalhar:", demand_df['Dia da Semana'].unique())
            
            day_df = demand_df[demand_df['Dia da Semana'] == selected_day]
            
            fig_day_demand = px.bar(day_df, x='Hora', y='chamadas_por_hora', title=f'Volume de Chamadas por Hora - {selected_day}',
                                    labels={'Hora': 'Hora do Dia', 'chamadas_por_hora': 'Volume de Chamadas'})
            st.plotly_chart(fig_day_demand, use_container_width=True)
            
            fig_day_tma = px.line(day_df, x='Hora', y='tma_medio', title=f'Variação do TMA por Hora - {selected_day}',
                                  labels={'Hora': 'Hora do Dia', 'tma_medio': 'TMA Médio (s)'}, markers=True)
            st.plotly_chart(fig_day_tma, use_container_width=True)

        # --- ABA 3: OTIMIZAÇÃO DE ESCALA ---
        with tab3:
            st.header("Comparativo de Escalas: Atual vs. Otimizada")
            
            if uploaded_current_scale is not None:
                current_scale_df = pd.read_csv(uploaded_current_scale, sep=None, engine='python')
                # Renomear colunas para consistência
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
                
                # Gráfico de Over/Understaffing
                fig_diff = px.bar(comparison_df, x='Hora', y='Diferença (Headcount)', color='Diferença (Headcount)',
                                  facet_row='Dia da Semana', height=800, title='Análise de Over/Understaffing por Hora',
                                  labels={'Diferença (Headcount)': 'Diferença de Headcount (Otimizado - Atual)'})
                st.plotly_chart(fig_diff, use_container_width=True)
            else:
                st.info("Faça o upload da sua escala atual na barra lateral para ver a análise comparativa e de custos.")
                st.subheader("Escala Otimizada Recomendada")
                st.dataframe(demand_df[['Dia da Semana', 'Hora', 'chamadas_por_hora', 'tma_medio', 'Escala Otimizada']].sort_values(by=['Dia da Semana', 'Hora']), hide_index=True)

        # --- ABA 4: SIMULADOR ---
        with tab4:
            st.header("Simulador de Cenários Pontuais")
            st.write("Faça testes 'E se...?' para prever o Nível de Serviço em diferentes cenários.")

            sim_col1, sim_col2, sim_col3 = st.columns(3)
            with sim_col1:
                sim_chamadas = st.number_input("Volume de Chamadas/Hora", min_value=1.0, value=50.0, step=1.0, key="sim_calls")
            with sim_col2:
                sim_tma = st.number_input("TMA (s)", min_value=1.0, value=180.0, step=5.0, key="sim_aht")
            with sim_col3:
                sim_atendentes = st.number_input("Nº de Atendentes", min_value=1, value=4, step=1, key="sim_agents")

            if st.button("Simular Cenário"):
                erlang = ErlangC(transactions=sim_chamadas, aht=sim_tma, interval=3600, asa=meta_tempo)
                sl_previsto = erlang.service_level(positions=sim_atendentes)
                ocupacao = erlang.achieved_occupancy(positions=sim_atendentes)
                
                st.subheader("Resultado da Simulação")
                res_col1, res_col2 = st.columns(2)
                res_col1.metric("Nível de Serviço Previsto", f"{sl_previsto:.2%}")
                res_col2.metric("Ocupação dos Atendentes", f"{ocupacao:.2%}")
                
                if sl_previsto >= meta_sl:
                    st.success("META ATINGIDA!")
                else:
                    st.warning("ALERTA! A meta não será atingida.")
else:
    st.info("Aguardando o upload do arquivo de histórico de chamadas para iniciar a análise.")