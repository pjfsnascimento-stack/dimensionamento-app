# ==============================================================================
# PLATAFORMA DE WORKFORCE MANAGEMENT (WFM) - VERSÃO 8.1 (CORREÇÃO DE LOCALE NA IA)
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
from prophet import Prophet
from prophet.plot import plot_plotly
from datetime import date

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="WFM Pro com IA - Otimizador de Escala",
    page_icon="🧠",
    layout="wide"
)

# --- DICIONÁRIO DE TRADUÇÃO GLOBAL ---
DAY_NAME_MAP_PT = {
    'Monday': 'Segunda-feira', 'Tuesday': 'Terça-feira', 'Wednesday': 'Quarta-feira',
    'Thursday': 'Quinta-feira', 'Friday': 'Sexta-feira', 'Saturday': 'Sábado', 'Sunday': 'Domingo'
}

# ==============================================================================
# MÓDULO DE PROCESSAMENTO DE DADOS (LÓGICA DE BACKEND)
# ==============================================================================
@st.cache_data
def process_uploaded_file(uploaded_file, column_mapping):
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
        raise ValueError("Nenhuma linha válida restou após a limpeza.")
    
    df['Dia da Semana'] = df['data_hora_inicio'].dt.day_name().map(DAY_NAME_MAP_PT)
    df['Hora'] = df['data_hora_inicio'].dt.hour
    return df, f"Arquivo processado. {len(df)} linhas válidas para análise."

@st.cache_data
def calculate_demand(_df, growth_forecast):
    df = _df.copy()
    df['year'] = df['data_hora_inicio'].dt.year
    df['week'] = df['data_hora_inicio'].dt.isocalendar().week
    num_semanas = df[['year', 'week']].drop_duplicates().shape[0] or 1
    demand_df = df.groupby(['Dia da Semana', 'Hora']).agg(tma_medio=('duracao_atendimento', 'mean'),total_chamadas=('data_hora_inicio', 'count')).reset_index()
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
# MÓDULO DE INTELIGÊNCIA ARTIFICIAL (MACHINE LEARNING)
# ==============================================================================

@st.cache_data
def prepare_data_for_prophet(_df):
    df_prophet = _df.set_index('data_hora_inicio').resample('H').size().reset_index(name='y')
    df_prophet.rename(columns={'data_hora_inicio': 'ds'}, inplace=True)
    return df_prophet

@st.cache_resource
def train_prophet_model(_df_prophet):
    model = Prophet(weekly_seasonality=True, daily_seasonality=True)
    model.fit(_df_prophet)
    return model

def generate_forecast(model, start_date, end_date):
    future_periods = (end_date - start_date).days * 24 + 23 # Garante que o último dia seja completo
    future = model.make_future_dataframe(periods=future_periods, freq='H')
    forecast = model.predict(future)
    forecast['ds'] = pd.to_datetime(forecast['ds'])
    mask = (forecast['ds'].dt.date >= start_date) & (forecast['ds'].dt.date <= end_date)
    return forecast[mask]

@st.cache_data
def generate_ai_staffing_schedule(_forecast, _historical_demand, meta_sl, meta_tempo, shrinkage, cost_per_hour):
    forecast_schedule = _forecast[['ds', 'yhat']].copy()
    forecast_schedule.rename(columns={'yhat': 'chamadas_previstas'}, inplace=True)
    forecast_schedule['chamadas_previstas'] = forecast_schedule['chamadas_previstas'].apply(lambda x: max(0, x))
    
    # CORREÇÃO DE LOCALE APLICADA AQUI
    forecast_schedule['Dia da Semana'] = forecast_schedule['ds'].dt.day_name().map(DAY_NAME_MAP_PT)
    forecast_schedule['Hora'] = forecast_schedule['ds'].dt.hour

    tma_map = _historical_demand.groupby(['Dia da Semana', 'Hora'])['tma_medio'].mean().reset_index()
    final_schedule = pd.merge(forecast_schedule, tma_map, on=['Dia da Semana', 'Hora'], how='left')
    final_schedule['tma_medio'].fillna(_historical_demand['tma_medio'].mean(), inplace=True)
    
    staff_needed = [_get_required_staff_binary_search(row['chamadas_previstas'], row['tma_medio'], meta_sl, meta_tempo) if row['chamadas_previstas'] > 0 else 0 for _, row in final_schedule.iterrows()]
    final_schedule['Atendentes Necessários'] = staff_needed
    final_schedule['Escala Prevista com IA'] = final_schedule['Atendentes Necessários'].apply(lambda x: 0 if x == 0 else math.ceil(x / (1 - shrinkage)))
    final_schedule['Custo Previsto (R$)'] = final_schedule['Escala Prevista com IA'] * cost_per_hour
    return final_schedule

# ==============================================================================
# CAMADA DE APRESENTAÇÃO (UI - STREAMLIT)
# ==============================================================================

st.title("🧠 WFM Pro com IA: Análise Preditiva de Escala")

with st.sidebar:
    st.image("https://i.imgur.com/l33C2F3.png", width=200)
    st.header("1. Upload de Dados")
    uploaded_history = st.file_uploader("A. Histórico de Chamadas (CSV)", type="csv", key="hist_uploader")
    column_mapping = {}
    if uploaded_history:
        header_df = pd.read_csv(uploaded_history, sep=None, engine='python', nrows=0)
        uploaded_history.seek(0)
        st.header("2. Mapeamento de Colunas")
        column_mapping['data_hora_inicio'] = st.selectbox("Data/Hora da Chamada", header_df.columns)
        column_mapping['duracao_atendimento'] = st.selectbox("Duração (em s)", header_df.columns)

    st.header("3. Metas e Parâmetros Globais")
    meta_sl = st.slider("Meta de Nível de Serviço (%)", 50, 100, 90) / 100.0
    meta_tempo = st.number_input("Tempo Alvo (s)", 1, value=15)
    cost_per_hour = st.number_input("Custo por Hora/Atendente (R$)", 1.0, value=25.0)
    shrinkage = st.slider("Shrinkage (%)", 0, 50, 25) / 100.0
    growth_forecast = st.slider("Crescimento de Volume (%)", -20, 100, 0)

if 'processed_data' not in st.session_state: st.session_state.processed_data = None
if 'demand_df' not in st.session_state: st.session_state.demand_df = None

if uploaded_history:
    if st.button("📊 Processar Dados Históricos", use_container_width=True, type="primary"):
        with st.spinner("Analisando histórico..."):
            try:
                df, info_message = process_uploaded_file(uploaded_history, column_mapping)
                st.session_state.processed_data = df
                st.session_state.demand_df = calculate_demand(df, growth_forecast)
                st.success(info_message)
            except Exception as e:
                st.error(f"Ocorreu um erro: {e}")
                st.session_state.processed_data = None; st.session_state.demand_df = None
else:
    st.info("Aguardando o upload do arquivo CSV para iniciar a análise.")

if st.session_state.demand_df is not None:
    demand_df_processed = st.session_state.demand_df
    tab1, tab2, tab3 = st.tabs([" dashboards Visão Geral", "💡 Otimização de Escala", "🧠 Previsão com IA"])

    with tab1:
        st.header("Dashboard de Performance Histórica")
        avg_calls = demand_df_processed['chamadas_por_hora'].mean()
        avg_tma = st.session_state.processed_data['duracao_atendimento'].mean()
        kpi1, kpi2 = st.columns(2)
        kpi1.metric("Média de Chamadas/Hora (Histórico)", f"{avg_calls:.1f}")
        kpi2.metric("TMA Médio Geral (Histórico)", f"{avg_tma:.1f}s")
        hourly_demand_agg = demand_df_processed.groupby('Hora', as_index=False)['chamadas_por_hora'].mean()
        fig_hourly = px.bar(hourly_demand_agg, x='Hora', y='chamadas_por_hora', title="Volume de Chamadas por Hora (Média Histórica)")
        st.plotly_chart(fig_hourly, use_container_width=True)

    with tab2:
        st.header("Escala Otimizada (Baseada no Histórico)")
        demand_df_processed['Escala Otimizada'] = calculate_required_staff(demand_df_processed, meta_sl, meta_tempo)
        demand_df_processed['Escala Otimizada com Shrinkage'] = demand_df_processed['Escala Otimizada'].apply(lambda x: 0 if x == 0 else math.ceil(x / (1 - shrinkage)))
        st.dataframe(demand_df_processed[['Dia da Semana', 'Hora', 'chamadas_por_hora', 'tma_medio', 'Escala Otimizada com Shrinkage']].sort_values(by=['Dia da Semana', 'Hora']), hide_index=True)

    with tab3:
        st.header("Geração de Escala Futura com Inteligência Artificial")
        st.info("O modelo de Machine Learning usará todo o histórico de chamadas para prever a demanda futura e recomendar a escala ótima.")
        
        today = date.today()
        col_ia1, col_ia2 = st.columns(2)
        start_date = col_ia1.date_input("Data de Início da Previsão", today)
        end_date = col_ia2.date_input("Data de Fim da Previsão", today + pd.Timedelta(days=6))

        if st.button("Gerar Previsão e Escala com IA", use_container_width=True, type="primary"):
            if start_date > end_date:
                st.error("A data de início deve ser anterior à data de fim.")
            else:
                with st.spinner("A IA está aprendendo com seus dados e gerando a previsão... Isso pode levar alguns minutos."):
                    try:
                        df_prophet = prepare_data_for_prophet(st.session_state.processed_data)
                        model = train_prophet_model(df_prophet)
                        forecast = generate_forecast(model, start_date, end_date)
                        final_schedule = generate_ai_staffing_schedule(forecast, demand_df_processed, meta_sl, meta_tempo, shrinkage, cost_per_hour)
                        st.success("Previsão e escala geradas com sucesso!")
                        
                        st.subheader("Gráfico da Previsão de Volume de Chamadas")
                        fig_forecast = plot_plotly(model, forecast)
                        fig_forecast.update_layout(title_text="Previsão de Chamadas por Hora (Histórico vs. Previsão da IA)", xaxis_title="Data", yaxis_title="Volume de Chamadas")
                        st.plotly_chart(fig_forecast, use_container_width=True)
                        
                        st.subheader("Escala Recomendada pela IA para o Período")
                        display_cols = ['ds', 'Dia da Semana', 'Hora', 'chamadas_previstas', 'tma_medio', 'Escala Prevista com IA', 'Custo Previsto (R$)']
                        final_schedule_display = final_schedule[display_cols].rename(columns={'ds': 'Data', 'tma_medio': 'TMA Previsto (s)'})
                        st.dataframe(final_schedule_display, hide_index=True)
                        
                        avg_forecasted_calls = final_schedule['chamadas_previstas'].mean()
                        total_forecasted_cost = final_schedule['Custo Previsto (R$)'].sum()
                        kpi_f1, kpi_f2 = st.columns(2)
                        kpi_f1.metric("Média de Chamadas/Hora (Previsto)", f"{avg_forecasted_calls:.1f}")
                        kpi_f2.metric(f"Custo Total Previsto para o Período", f"R$ {total_forecasted_cost:,.2f}")
                    except Exception as e:
                        st.error(f"Ocorreu um erro durante a previsão com IA: {e}")