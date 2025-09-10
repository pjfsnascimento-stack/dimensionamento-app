# ==============================================================================
# PLATAFORMA DE WORKFORCE MANAGEMENT (WFM) - VERSÃO 11.0 (UX APRIMORADA)
# Implementa tooltips, gráficos didáticos e tabelas profissionais.
# ==============================================================================
import streamlit as st
import pandas as pd
from pandas.errors import ParserError
import plotly.express as px
import plotly.graph_objects as go
import math
from pyworkforce.queuing import ErlangC
import io
import csv
import chardet
from prophet import Prophet
from prophet.plot import plot_plotly
from datetime import date
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import Font, Alignment

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="WFM Pro AI - Análise Preditiva",
    page_icon="⭐",
    layout="wide"
)

# ==============================================================================
# MÓDULOS DE LÓGICA DE NEGÓCIO (BACKEND)
# ==============================================================================
@st.cache_data
def to_excel(data_dict: dict) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheet_name, data in data_dict.items():
            if isinstance(data, pd.DataFrame):
                data.to_excel(writer, sheet_name=sheet_name, index=False)
            else:
                df_text = pd.DataFrame({'Relatorio': [data]})
                df_text.to_excel(writer, sheet_name=sheet_name, index=False)
                worksheet = writer.sheets[sheet_name]
                worksheet.column_dimensions['A'].width = 100
                cell = worksheet['A2']
                cell.alignment = Alignment(wrap_text=True, vertical='top')
    return output.getvalue()

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
    # Sanitização: Remove durações negativas ou excessivamente longas (ex: > 4 horas)
    df = df[(df['duracao_atendimento'] >= 0) & (df['duracao_atendimento'] < 14400)]
    df.dropna(subset=['data_hora_inicio', 'duracao_atendimento'], inplace=True)
    if df.empty:
        raise ValueError("Nenhuma linha válida restou após a limpeza.")
    day_name_map_pt = {'Monday': 'Segunda-feira', 'Tuesday': 'Terça-feira', 'Wednesday': 'Quarta-feira','Thursday': 'Quinta-feira', 'Friday': 'Sexta-feira', 'Saturday': 'Sábado', 'Sunday': 'Domingo'}
    df['Dia da Semana'] = df['data_hora_inicio'].dt.day_name().map(day_name_map_pt)
    df['Hora'] = df['data_hora_inicio'].dt.hour
    return df, f"Arquivo processado com sucesso. {len(df)} linhas válidas para análise."

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
    return [_get_required_staff_binary_search(row['chamadas_por_hora'], row['tma_medio'], meta_sl, meta_tempo) for _, row in _demand_df.iterrows()]

# ... (Funções de IA permanecem as mesmas) ...
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

@st.cache_data
def generate_ai_report(_forecast_df):
    forecast_df = _forecast_df.copy()
    forecast_df.rename(columns={'yhat': 'chamadas_previstas'}, inplace=True)
    day_name_map_pt = {'Monday': 'Segunda-feira', 'Tuesday': 'Terça-feira', 'Wednesday': 'Quarta-feira','Thursday': 'Quinta-feira', 'Friday': 'Sexta-feira', 'Saturday': 'Sábado', 'Sunday': 'Domingo'}
    forecast_df['Dia da Semana'] = forecast_df['ds'].dt.day_name().map(day_name_map_pt)
    forecast_df['Hora'] = forecast_df['ds'].dt.hour
    overall_trend = forecast_df['trend'].iloc[-1] - forecast_df['trend'].iloc[0]
    trend_text = "de crescimento estável" if abs(overall_trend) < 1 else "de forte crescimento" if overall_trend > 0 else "de forte declínio"
    daily_summary = forecast_df.groupby('Dia da Semana')['chamadas_previstas'].sum().sort_values(ascending=False)
    peak_day = daily_summary.index[0]
    hourly_summary = forecast_df.groupby('Hora')['chamadas_previstas'].mean().sort_values(ascending=False)
    peak_hour = hourly_summary.index[0]
    report = f"""
    ### Relatório Executivo da Previsão de IA
    **1. Análise de Tendência Geral:** A análise indica uma tendência **{trend_text}** no volume de chamadas.
    **2. Padrões de Demanda:** O dia de maior volume previsto é **{peak_day}**, com o pico de demanda ocorrendo por volta das **{peak_hour}:00**.
    **3. Recomendações:** Foque seus recursos no período de pico e utilize horários de baixo volume para tarefas secundárias. Use a tabela detalhada abaixo para construir sua grade horária.
    """
    return report

# ==============================================================================
# CAMADA DE APRESENTAÇÃO (UI - STREAMLIT)
# ==============================================================================

st.title("⭐ WFM Pro AI: Análise e Otimização Inteligente")

# --- ÁREA DE UPLOAD E MAPEAMENTO (Melhoria 1) ---
with st.container(border=True):
    st.header("1. Carregue seu Histórico de Chamadas")
    st.markdown("Comece fazendo o upload do seu arquivo de dados no formato `.csv`. A ferramenta é flexível e permitirá que você mapeie as colunas do seu arquivo para os campos necessários.")
    
    uploaded_history = st.file_uploader(
        "Selecione o arquivo CSV", 
        type="csv", 
        label_visibility="collapsed"
    )
    
    # Inicializa o estado da sessão para armazenar os dados processados
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None

    if uploaded_history:
        try:
            header_df = pd.read_csv(uploaded_history, sep=None, engine='python', nrows=0)
            uploaded_history.seek(0)
            
            st.subheader("Mapeamento de Colunas")
            st.write("Associe as colunas do seu arquivo aos campos que a plataforma precisa para a análise.")
            map_cols = st.columns(2)
            column_mapping = {
                'data_hora_inicio': map_cols[0].selectbox("Coluna de Data/Hora da Chamada", header_df.columns, help="Selecione a coluna que contém o timestamp exato de cada chamada."),
                'duracao_atendimento': map_cols[1].selectbox("Coluna de Duração (em segundos)", header_df.columns, help="Selecione a coluna com a duração do atendimento em segundos.")
            }
            
            if st.button("📊 Processar e Analisar Dados", use_container_width=True, type="primary"):
                st.session_state.processed_data, st.session_state.info_message = process_uploaded_file(uploaded_history, column_mapping)
                st.success(st.session_state.info_message)
        except Exception as e:
            st.error(f"Não foi possível ler o cabeçalho do arquivo. Verifique se é um CSV válido. Erro: {e}")

# --- ABAS DE RESULTADOS (só aparecem após o processamento) ---
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
        hourly_agg = demand_df_processed.groupby('Hora', as_index=False).agg({
            'chamadas_por_hora': 'mean',
            'tma_medio': 'mean',
            'desvio_padrao_tma': 'mean'
        })
        fig = go.Figure()
        fig.add_trace(go.Bar(x=hourly_agg['Hora'], y=hourly_agg['chamadas_por_hora'], name='Volume de Chamadas', yaxis='y1'))
        fig.add_trace(go.Scatter(x=hourly_agg['Hora'], y=hourly_agg['tma_medio'], name='TMA Médio (s)', yaxis='y2', mode='lines+markers'))
        fig.update_layout(
            title_text="<b>Volume de Chamadas vs. TMA por Hora (Média Semanal)</b>",
            template="plotly_white",
            yaxis=dict(title='Volume Médio de Chamadas'),
            yaxis2=dict(title='TMA Médio (s)', overlaying='y', side='right'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Analise a correlação entre o volume de chamadas (barras) e o tempo médio de atendimento (linha) ao longo do dia.")

    with tab2:
        st.header("Otimização de Escala (Baseada no Histórico)")
        st.write("Ajuste suas metas operacionais para receber a recomendação de escala otimizada, apresentada de forma visual e interativa.")
        
        opt_cols = st.columns(3)
        meta_sl = opt_cols[0].slider("Meta de Nível de Serviço (%)", 50, 100, 90, help="O percentual de chamadas que devem ser atendidas dentro do tempo alvo. Ex: 90%.") / 100.0
        meta_tempo = opt_cols[1].number_input("Tempo Alvo de Atendimento (s)", 1, value=15, help="O tempo máximo (em segundos) que uma chamada deve esperar na fila para a meta de NS ser cumprida.")
        shrinkage = opt_cols[2].slider("Fator de Shrinkage (%)", 0, 50, 25, help="O percentual de tempo em que um atendente está pago, mas não disponível (pausas, reuniões, etc.).") / 100.0
        
        demand_df_processed['Atendentes na Escala'] = calculate_required_staff(demand_df_processed, meta_sl, meta_tempo)
        demand_df_processed['Atendentes na Escala'] = demand_df_processed['Atendentes na Escala'].apply(lambda x: 0 if x == 0 else math.ceil(x / (1 - shrinkage)))
        
        st.subheader("Tabela de Escala Profissional e Interativa")
        tabela_formatada = demand_df_processed.rename(columns={'chamadas_por_hora': 'Volume Médio/Hora', 'tma_medio': 'TMA Médio (s)', 'Atendentes na Escala': 'Atendentes na Escala'})
        st.dataframe(
            tabela_formatada[['Dia da Semana', 'Hora', 'Volume Médio/Hora', 'TMA Médio (s)', 'Atendentes na Escala']].style
            .format({'Volume Médio/Hora': "{:.1f}", 'TMA Médio (s)': "{:.0f}"})
            .background_gradient(cmap='viridis', subset=['Volume Médio/Hora'])
            .bar(subset=['Atendentes na Escala'], color='#5fba7d', align='zero'),
            hide_index=True, use_container_width=True
        )
        st.caption("As cores na coluna 'Volume' indicam os horários de maior demanda. As barras na coluna 'Atendentes' facilitam a visualização da dimensão da equipe necessária.")

        excel_data_optimized = to_excel({'Escala Otimizada': tabela_formatada[['Dia da Semana', 'Hora', 'Volume Médio/Hora', 'TMA Médio (s)', 'Atendentes na Escala']]})
        st.download_button("📥 Exportar para Excel (.xlsx)", excel_data_optimized, "escala_otimizada.xlsx")


    with tab3:
        st.header("Geração de Escala Futura com Inteligência Artificial")
        # ... (código da aba de IA da v10.1, que já está bem estruturado) ...
        today = date.today()
        ia_cols = st.columns(2)
        start_date = ia_cols[0].date_input("Data de Início da Previsão", today)
        end_date = ia_cols[1].date_input("Data de Fim da Previsão", today + pd.Timedelta(days=6))
        if st.button("🚀 Gerar Previsão e Escala com IA", use_container_width=True, type="primary"):
            with st.spinner("A IA está aprendendo com seus dados e gerando a previsão... Isso pode levar alguns minutos."):
                df_prophet = prepare_data_for_prophet(st.session_state.processed_data)
                model = train_prophet_model(df_prophet)
                forecast = generate_forecast(model, start_date, end_date)
                final_schedule = generate_ai_staffing_schedule(forecast, demand_df_processed, meta_sl, meta_tempo, shrinkage)
                ai_report = generate_ai_report(forecast)
                st.success("Previsão e escala geradas com sucesso!")
                st.subheader("Relatório Executivo da IA")
                st.markdown(ai_report)
                st.subheader("Gráfico Interativo da Previsão")
                fig_forecast = plot_plotly(model, forecast)
                fig_forecast.update_layout(title_text="<b>Previsão de Chamadas por Hora (Histórico vs. Previsão da IA)</b>", xaxis_title="Data", yaxis_title="Volume de Chamadas")
                st.plotly_chart(fig_forecast, use_container_width=True)
                st.caption("A área azul representa o intervalo de confiança da previsão. Quanto mais larga, maior a incerteza do modelo.")
                st.subheader("Escala Detalhada Recomendada pela IA")
                display_cols = ['ds', 'Dia da Semana', 'Hora', 'chamadas_previstas', 'tma_medio', 'Escala Prevista com IA']
                final_schedule_display = final_schedule[display_cols].rename(columns={'ds': 'Data', 'tma_medio': 'TMA Previsto (s)'})
                st.dataframe(final_schedule_display, hide_index=True)
                
                excel_data_ai = to_excel({'Relatorio_IA': ai_report, 'Escala_Prevista_IA': final_schedule_display})
                st.download_button("📥 Exportar Relatório e Escala da IA para Excel (.xlsx)", excel_data_ai, "previsao_escala_ia.xlsx")