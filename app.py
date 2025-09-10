# ==============================================================================
# PLATAFORMA DE WORKFORCE MANAGEMENT (WFM) - VERSÃO 9.0 (EXPERIÊNCIA DO USUÁRIO APRIMORADA)
# Implementa parâmetros contextuais, gráficos didáticos e tabelas profissionais.
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
    page_icon="⭐",
    layout="wide"
)

# ==============================================================================
# MÓDULO DE PROCESSAMENTO DE DADOS (LÓGICA DE BACKEND)
# ==============================================================================
# ... (As funções de backend permanecem as mesmas, pois já são robustas) ...
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
    day_name_map_pt = {'Monday': 'Segunda-feira', 'Tuesday': 'Terça-feira', 'Wednesday': 'Quarta-feira','Thursday': 'Quinta-feira', 'Friday': 'Sexta-feira', 'Saturday': 'Sábado', 'Sunday': 'Domingo'}
    df['Dia da Semana'] = df['data_hora_inicio'].dt.day_name().map(day_name_map_pt)
    df['Hora'] = df['data_hora_inicio'].dt.hour
    return df, f"Arquivo processado. {len(df)} linhas válidas para análise."

@st.cache_data
def calculate_demand(_df):
    df = _df.copy()
    df['year'] = df['data_hora_inicio'].dt.year
    df['week'] = df['data_hora_inicio'].dt.isocalendar().week
    num_semanas = df[['year', 'week']].drop_duplicates().shape[0] or 1
    demand_df = df.groupby(['Dia da Semana', 'Hora']).agg(
        tma_medio=('duracao_atendimento', 'mean'),
        desvio_padrao_tma=('duracao_atendimento', 'std'),
        total_chamadas=('data_hora_inicio', 'count')
    ).reset_index().fillna(0)
    demand_df['chamadas_por_hora'] = (demand_df['total_chamadas'] / num_semanas)
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
# ... (demais funções de IA) ...
def generate_forecast(model, start_date, end_date):
    future_periods = (end_date - start_date).days * 24 + 23
    future = model.make_future_dataframe(periods=future_periods, freq='H')
    forecast = model.predict(future)
    forecast['ds'] = pd.to_datetime(forecast['ds'])
    mask = (forecast['ds'].dt.date >= start_date) & (forecast['ds'].dt.date <= end_date)
    return forecast[mask]

@st.cache_data
def generate_ai_staffing_schedule(_forecast, _historical_demand, meta_sl, meta_tempo, shrinkage):
    forecast_schedule = _forecast[['ds', 'yhat']].copy()
    forecast_schedule.rename(columns={'yhat': 'chamadas_previstas'}, inplace=True)
    forecast_schedule['chamadas_previstas'] = forecast_schedule['chamadas_previstas'].apply(lambda x: max(0, x))
    day_name_map_pt = {'Monday': 'Segunda-feira', 'Tuesday': 'Terça-feira', 'Wednesday': 'Quarta-feira','Thursday': 'Quinta-feira', 'Friday': 'Sexta-feira', 'Saturday': 'Sábado', 'Sunday': 'Domingo'}
    forecast_schedule['Dia da Semana'] = forecast_schedule['ds'].dt.day_name().map(day_name_map_pt)
    forecast_schedule['Hora'] = forecast_schedule['ds'].dt.hour
    tma_map = _historical_demand.groupby(['Dia da Semana', 'Hora'])['tma_medio'].mean().reset_index()
    final_schedule = pd.merge(forecast_schedule, tma_map, on=['Dia da Semana', 'Hora'], how='left')
    final_schedule['tma_medio'].fillna(_historical_demand['tma_medio'].mean(), inplace=True)
    staff_needed = [_get_required_staff_binary_search(row['chamadas_previstas'], row['tma_medio'], meta_sl, meta_tempo) if row['chamadas_previstas'] > 0 else 0 for _, row in final_schedule.iterrows()]
    final_schedule['Atendentes Necessários'] = staff_needed
    final_schedule['Escala Prevista com IA'] = final_schedule['Atendentes Necessários'].apply(lambda x: 0 if x == 0 else math.ceil(x / (1 - shrinkage)))
    return final_schedule

# ==============================================================================
# CAMADA DE APRESENTAÇÃO (UI - STREAMLIT)
# ==============================================================================

st.title("⭐ WFM Pro com IA: Análise e Otimização Inteligente")

# --- ÁREA DE UPLOAD E MAPEAMENTO ---
with st.container(border=True):
    st.header("1. Carregue seu Histórico de Chamadas")
    st.write("Comece fazendo o upload do seu arquivo de dados no formato .csv. A ferramenta irá guiá-lo no mapeamento das colunas necessárias.")
    
    uploaded_history = st.file_uploader("Selecione o arquivo CSV", type="csv", label_visibility="collapsed")
    
    column_mapping = {}
    if uploaded_history:
        header_df = pd.read_csv(uploaded_history, sep=None, engine='python', nrows=0)
        uploaded_history.seek(0)
        
        st.subheader("Mapeamento de Colunas")
        st.write("Associe as colunas do seu arquivo aos campos que a plataforma precisa para a análise.")
        map_cols = st.columns(2)
        column_mapping['data_hora_inicio'] = map_cols[0].selectbox("Coluna de Data/Hora da Chamada", header_df.columns)
        column_mapping['duracao_atendimento'] = map_cols[1].selectbox("Coluna de Duração (em segundos)", header_df.columns)
        
        if st.button("📊 Processar e Analisar Dados", use_container_width=True, type="primary"):
            st.session_state.processed_data, st.session_state.info_message = process_uploaded_file(uploaded_history, column_mapping)
            st.success(st.session_state.info_message)

# --- INICIALIZAÇÃO DO ESTADO ---
if 'processed_data' not in st.session_state: st.session_state.processed_data = None

# --- ABAS DE RESULTADOS ---
if st.session_state.processed_data is not None:
    demand_df_processed = calculate_demand(st.session_state.processed_data)
    
    tab1, tab2, tab3 = st.tabs([" dashboards Visão Geral", "💡 Otimização de Escala", "🧠 Previsão com IA"])

    with tab1:
        st.header("Dashboard de Performance Histórica")
        avg_calls = demand_df_processed['chamadas_por_hora'].mean()
        avg_tma = st.session_state.processed_data['duracao_atendimento'].mean()
        kpi1, kpi2 = st.columns(2)
        kpi1.metric("Média de Chamadas/Hora (Histórico)", f"{avg_calls:.1f}")
        kpi2.metric("TMA Médio Geral (Histórico)", f"{avg_tma:.1f}s")
        
        st.subheader("Análise Interativa: Volume de Chamadas e TMA por Hora")
        hourly_demand_agg = demand_df_processed.groupby('Hora', as_index=False).agg({
            'chamadas_por_hora': 'mean',
            'tma_medio': 'mean',
            'desvio_padrao_tma': 'mean'
        })
        fig_hourly = px.bar(
            hourly_demand_agg, x='Hora', y='chamadas_por_hora',
            title="<b>Volume de Chamadas e TMA por Hora (Média Semanal)</b>",
            labels={'chamadas_por_hora': 'Volume Médio de Chamadas', 'Hora': 'Hora do Dia'},
            hover_data={'tma_medio': ':.0f', 'desvio_padrao_tma': ':.1f'},
            template="plotly_white"
        )
        fig_hourly.update_layout(title_font_size=20)
        st.plotly_chart(fig_hourly, use_container_width=True)
        st.caption("Passe o mouse sobre as barras para ver o TMA médio e o desvio padrão para cada hora.")

    with tab2:
        st.header("Otimização de Escala (Baseada no Histórico)")
        st.write("Ajuste suas metas operacionais para receber a recomendação de escala otimizada.")
        
        opt_cols = st.columns(3)
        meta_sl = opt_cols[0].slider("Meta de Nível de Serviço (%)", 50, 100, 90, help="O percentual de chamadas que devem ser atendidas dentro do tempo alvo.") / 100.0
        meta_tempo = opt_cols[1].number_input("Tempo Alvo de Atendimento (s)", 1, value=15, help="O tempo máximo (em segundos) que uma chamada deve esperar na fila.")
        shrinkage = opt_cols[2].slider("Fator de Shrinkage (%)", 0, 50, 25, help="O percentual de tempo em que um atendente está pago, mas não disponível (pausas, reuniões, etc.).") / 100.0
        
        demand_df_processed['Atendentes Necessários'] = calculate_required_staff(demand_df_processed, meta_sl, meta_tempo)
        demand_df_processed['Escala Otimizada'] = demand_df_processed['Atendentes Necessários'].apply(lambda x: 0 if x == 0 else math.ceil(x / (1 - shrinkage)))
        
        st.subheader("Tabela de Escala Profissional")
        tabela_formatada = demand_df_processed.rename(columns={'chamadas_por_hora': 'Volume Médio/Hora', 'tma_medio': 'TMA Médio (s)', 'Escala Otimizada': 'Atendentes na Escala'})
        st.dataframe(
            tabela_formatada[['Dia da Semana', 'Hora', 'Volume Médio/Hora', 'TMA Médio (s)', 'Atendentes na Escala']].style
            .format({'Volume Médio/Hora': "{:.1f}", 'TMA Médio (s)': "{:.0f}"})
            .background_gradient(cmap='viridis', subset=['Volume Médio/Hora'])
            .bar(subset=['Atendentes na Escala'], color='#5fba7d', align='zero'),
            hide_index=True, use_container_width=True
        )
        st.caption("As cores na coluna 'Volume' indicam os horários de maior demanda. As barras na coluna 'Atendentes' facilitam a visualização da dimensão da equipe necessária.")

    with tab3:
        st.header("Geração de Escala Futura com Inteligência Artificial")
        st.write("Selecione um período futuro para que o modelo de Machine Learning preveja a demanda e recomende a escala ótima.")
        
        today = date.today()
        ia_cols = st.columns(2)
        start_date = ia_cols[0].date_input("Data de Início da Previsão", today)
        end_date = ia_cols[1].date_input("Data de Fim da Previsão", today + pd.Timedelta(days=6))

        if st.button("Gerar Previsão e Escala com IA", use_container_width=True, type="primary"):
            if start_date > end_date:
                st.error("A data de início deve ser anterior à data de fim.")
            else:
                with st.spinner("A IA está aprendendo com seus dados e gerando a previsão... Isso pode levar alguns minutos."):
                    df_prophet = prepare_data_for_prophet(st.session_state.processed_data)
                    model = train_prophet_model(df_prophet)
                    forecast = generate_forecast(model, start_date, end_date)
                    final_schedule = generate_ai_staffing_schedule(forecast, demand_df_processed, meta_sl, meta_tempo, shrinkage)
                    
                    st.success("Previsão e escala geradas com sucesso!")
                    st.subheader("Gráfico da Previsão de Volume de Chamadas")
                    fig_forecast = plot_plotly(model, forecast)
                    fig_forecast.update_layout(title_text="Previsão de Chamadas por Hora (Histórico vs. Previsão da IA)", xaxis_title="Data", yaxis_title="Volume de Chamadas")
                    st.plotly_chart(fig_forecast, use_container_width=True)
                    st.subheader("Escala Recomendada pela IA para o Período")
                    st.dataframe(final_schedule[['ds', 'Dia da Semana', 'Hora', 'chamadas_previstas', 'tma_medio', 'Escala Prevista com IA']].rename(columns={'ds': 'Data', 'tma_medio': 'TMA Previsto (s)'}), hide_index=True)