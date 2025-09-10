# ==============================================================================
# PLATAFORMA DE WORKFORCE MANAGEMENT (WFM) - VERSÃO 6.0 (SIMULAÇÃO AVANÇADA)
# Reorganiza a UI e introduz um simulador de múltiplos cenários.
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

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="WFM Pro - Otimizador de Escala",
    page_icon="⚡️",
    layout="wide"
)

# ==============================================================================
# MÓDULO DE PROCESSAMENTO DE DADOS (LÓGICA DE BACKEND)
# ==============================================================================

@st.cache_data
def process_uploaded_file(uploaded_file, column_mapping):
    """Lê, valida e prepara o arquivo de histórico de chamadas."""
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
        raise ValueError(f"Mapeamento incompleto: faltam {required_columns - set(df.columns)}")

    df['data_hora_inicio'] = pd.to_datetime(df['data_hora_inicio'], format='%Y:%m:%d %H:%M:%S', errors='coerce')
    df['duracao_atendimento'] = pd.to_numeric(df['duracao_atendimento'], errors='coerce')
    
    df.dropna(subset=['data_hora_inicio', 'duracao_atendimento'], inplace=True)
    if df.empty:
        raise ValueError("Nenhuma linha válida restou após a limpeza. Verifique o formato das datas e durações.")

    day_name_map_pt = {
        'Monday': 'Segunda-feira', 'Tuesday': 'Terça-feira', 'Wednesday': 'Quarta-feira',
        'Thursday': 'Quinta-feira', 'Friday': 'Sexta-feira', 'Saturday': 'Sábado', 'Sunday': 'Domingo'
    }
    df['Dia da Semana'] = df['data_hora_inicio'].dt.day_name().map(day_name_map_pt)
    df['Hora'] = df['data_hora_inicio'].dt.hour
    
    return df, f"Arquivo processado. {len(df)} linhas válidas para análise."

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
def _get_required_staff_binary_search(chamadas, tma, meta_sl, meta_tempo):
    low, high, best_n = 1, 100, 100
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

st.title("⚡️ WFM Pro: Plataforma Analítica com Simulação Avançada")

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
        column_mapping['duracao_atendimento'] = st.selectbox("Duração (em s)", header_df.columns)
        column_mapping['Fila/Cliente'] = st.selectbox("Fila/Cliente", [None] + list(header_df.columns))

    st.header("3. Metas e Parâmetros Globais")
    meta_sl = st.slider("Meta de Nível de Serviço (%)", 50, 100, 90) / 100.0
    meta_tempo = st.number_input("Tempo Alvo de Atendimento (s)", 1, value=15)
    cost_per_hour = st.number_input("Custo por Hora/Atendente (R$)", 1.0, value=25.0)
    shrinkage = st.slider("Fator de Shrinkage (%)", 0, 50, 25) / 100.0
    growth_forecast = st.slider("Previsão de Crescimento de Volume (%)", -20, 100, 0)

# --- INICIALIZAÇÃO E LÓGICA PRINCIPAL ---
if 'processed_data' not in st.session_state: st.session_state.processed_data = None
if 'demand_df' not in st.session_state: st.session_state.demand_df = None

if uploaded_history:
    if st.button("📊 Processar e Analisar Dados", use_container_width=True, type="primary"):
        with st.spinner("Analisando histórico..."):
            try:
                df, info_message = process_uploaded_file(uploaded_history, column_mapping)
                st.session_state.processed_data = df
                st.session_state.demand_df = calculate_demand(df, growth_forecast)
                st.success(info_message)
            except Exception as e:
                st.error(f"Ocorreu um erro: {e}")
                st.session_state.processed_data = None
                st.session_state.demand_df = None
else:
    st.info("Aguardando o upload do arquivo CSV para iniciar a análise.")

# --- ABAS DE RESULTADOS (só aparecem se os dados foram processados) ---
if st.session_state.demand_df is not None:
    demand_df_processed = st.session_state.demand_df
    
    demand_df_processed['Atendentes Necessários'] = calculate_required_staff(demand_df_processed, meta_sl, meta_tempo)
    demand_df_processed['Escala Otimizada'] = demand_df_processed['Atendentes Necessários'].apply(lambda x: 0 if x == 0 else math.ceil(x / (1 - shrinkage)))
    
    tab1, tab2, tab3 = st.tabs([" dashboards Visão Geral", "💡 Otimização de Escala", "🔮 Previsão e Simulação Avançada"])

    with tab1:
        st.header("Dashboard de Performance da Operação")
        avg_calls = demand_df_processed['chamadas_por_hora'].mean()
        avg_tma = st.session_state.processed_data['duracao_atendimento'].mean()
        kpi1, kpi2 = st.columns(2)
        kpi1.metric("Média de Chamadas/Hora", f"{avg_calls:.1f}")
        kpi2.metric("TMA Médio Geral", f"{avg_tma:.1f}s")

        hourly_demand_agg = demand_df_processed.groupby('Hora', as_index=False)['chamadas_por_hora'].mean()
        fig_hourly = px.bar(hourly_demand_agg, x='Hora', y='chamadas_por_hora', title="Volume de Chamadas por Hora do Dia (Média Semanal)")
        st.plotly_chart(fig_hourly, use_container_width=True)

    with tab2:
        st.header("Escala Otimizada Recomendada")
        st.dataframe(demand_df_processed[['Dia da Semana', 'Hora', 'chamadas_por_hora', 'tma_medio', 'Escala Otimizada']].sort_values(by=['Dia da Semana', 'Hora']), hide_index=True)

    with tab3:
        st.header("Simulador de Múltiplos Cenários")
        st.write("Selecione múltiplos dias e horas para simular o impacto de uma escala em um turno ou período completo.")
        
        dias_disponiveis = list(demand_df_processed['Dia da Semana'].unique())
        horas_disponiveis = sorted(list(demand_df_processed['Hora'].unique()))

        sim_col1, sim_col2 = st.columns(2)
        dias_selecionados = sim_col1.multiselect("Selecione os Dias da Semana", dias_disponiveis, default=dias_disponiveis[:5])
        horas_selecionadas = sim_col2.multiselect("Selecione as Horas do Dia", horas_disponiveis, default=[h for h in horas_disponiveis if 8 <= h <= 18])
        
        atendentes_sim = st.slider("Nº de Atendentes para Simular nos horários selecionados", 1, 50, 5)

        if st.button("Simular Múltiplos Cenários", type="primary"):
            cenarios = demand_df_processed[(demand_df_processed['Dia da Semana'].isin(dias_selecionados)) & (demand_df_processed['Hora'].isin(horas_selecionadas))]
            
            if cenarios.empty:
                st.warning("Nenhum dado encontrado para os dias e horas selecionados.")
            else:
                resultados = []
                for _, row in cenarios.iterrows():
                    vol_real, tma_real = row['chamadas_por_hora'], row['tma_medio']
                    if vol_real > 0 and tma_real > 0:
                        erlang = ErlangC(transactions=vol_real, aht=tma_real, interval=3600, asa=meta_tempo)
                        sl_previsto = erlang.service_level(positions=atendentes_sim)
                        ocupacao = erlang.achieved_occupancy(positions=atendentes_sim)
                        resultados.append({'Dia': row['Dia da Semana'], 'Hora': row['Hora'], 'Volume Esperado': f"{vol_real:.1f}", 'TMA Esperado (s)': f"{tma_real:.1f}", 'SL Previsto': sl_previsto, 'Ocupação Prevista': ocupacao})
                
                resultados_df = pd.DataFrame(resultados)
                
                st.subheader("Resultados da Simulação em Lote")
                
                # KPI de resumo
                sl_medio_periodo = resultados_df['SL Previsto'].mean()
                st.metric(f"Nível de Serviço Médio para os {len(resultados_df)} cenários simulados", f"{sl_medio_periodo:.2%}")

                # Tabela com cores
                def color_sl(val):
                    color = 'green' if val >= meta_sl else 'red'
                    return f'color: {color}; font-weight: bold;'
                
                st.dataframe(resultados_df.style.applymap(color_sl, subset=['SL Previsto']).format({'SL Previsto': "{:.2%}", 'Ocupação Prevista': "{:.2%}"}), use_container_width=True)

# --- SIMULADOR RÁPIDO INDEPENDENTE (SEMPRE VISÍVEL) ---
st.header("🔬 Simulador Rápido (Independente)")
st.write("Faça testes rápidos para qualquer cenário hipotético, sem precisar de um arquivo.")
sim_ind_col1, sim_ind_col2, sim_ind_col3 = st.columns(3)
sim_ind_chamadas = sim_ind_col1.number_input("Volume de Chamadas/Hora", 1.0, value=50.0, step=1.0, key="sim_ind_calls")
sim_ind_tma = sim_ind_col2.number_input("TMA (s)", 1.0, value=180.0, step=5.0, key="sim_ind_aht")
sim_ind_atendentes = sim_ind_col3.number_input("Nº de Atendentes", 1, value=4, step=1, key="sim_ind_agents")
if st.button("Simular Cenário Rápido"):
    erlang_ind = ErlangC(transactions=sim_ind_chamadas, aht=sim_ind_tma, interval=3600, asa=meta_tempo)
    sl_ind, ocupacao_ind = erlang_ind.service_level(positions=sim_ind_atendentes), erlang_ind.achieved_occupancy(positions=sim_ind_atendentes)
    st.subheader("Resultado da Simulação Rápida")
    res_ind_col1, res_ind_col2 = st.columns(2)
    res_ind_col1.metric("Nível de Serviço Previsto", f"{sl_ind:.2%}")
    res_ind_col2.metric("Ocupação dos Atendentes", f"{ocupacao_ind:.2%}")
    if sl_ind >= meta_sl: st.success("META ATINGIDA!")
    else: st.warning("ALERTA! A meta não será atingida.")