# ==============================================================================
# PLATAFORMA DE WORKFORCE MANAGEMENT (WFM) - VERSÃO 10.0 (EDIÇÃO ANALÍTICA)
# Implementa relatórios de IA em texto, gráficos profissionais e exportação para Excel.
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
    page_icon="🏆",
    layout="wide"
)

# ==============================================================================
# MÓDULO DE HELPERS E EXPORTAÇÃO
# ==============================================================================

@st.cache_data
def to_excel(data_dict: dict) -> bytes:
    """Converte um dicionário de DataFrames para um arquivo Excel em memória."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheet_name, data in data_dict.items():
            if isinstance(data, pd.DataFrame):
                data.to_excel(writer, sheet_name=sheet_name, index=False)
            else: # Para relatórios de texto
                df_text = pd.DataFrame({'Relatorio': [data]})
                df_text.to_excel(writer, sheet_name=sheet_name, index=False)
                # Ajusta a formatação da planilha de texto
                worksheet = writer.sheets[sheet_name]
                worksheet.column_dimensions['A'].width = 100
                cell = worksheet['A2']
                cell.alignment = Alignment(wrap_text=True, vertical='top')
    processed_data = output.getvalue()
    return processed_data

# ==============================================================================
# MÓDULO DE PROCESSAMENTO DE DADOS (LÓGICA DE BACKEND)
# ==============================================================================
# ... (Funções de processamento, cálculo e IA permanecem as mesmas da v9.0) ...
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
    """Gera um relatório textual com insights da previsão da IA."""
    overall_trend = _forecast_df['trend'].iloc[-1] - _forecast_df['trend'].iloc[0]
    trend_text = "de crescimento estável" if abs(overall_trend) < 1 else "de forte crescimento" if overall_trend > 0 else "de forte declínio"
    
    daily_summary = _forecast_df.groupby('Dia da Semana')['chamadas_previstas'].sum().sort_values(ascending=False)
    peak_day = daily_summary.index[0]
    
    hourly_summary = _forecast_df.groupby('Hora')['chamadas_previstas'].mean().sort_values(ascending=False)
    peak_hour = hourly_summary.index[0]
    
    report = f"""
    ### Relatório Executivo da Previsão de IA

    **1. Análise de Tendência Geral:**
    A análise do histórico indica uma tendência **{trend_text}** no volume de chamadas para o período previsto. É crucial preparar a operação para esta trajetória.

    **2. Padrões de Demanda Semanais:**
    - **Dia de Maior Volume:** O modelo prevê que **{peak_day}** será o dia com maior concentração de chamadas. A escala para este dia deve ser robusta.
    - **Horário de Pico:** O horário de maior demanda, em média, será às **{peak_hour}:00**. Este é um ponto crítico que exige atenção máxima na alocação de pessoal.

    **3. Insights e Recomendações:**
    - **Foco no Pico:** Concentre seus atendentes mais experientes durante o período de pico ({peak_day} por volta das {peak_hour}:00) para manter a eficiência e o Nível de Serviço.
    - **Oportunidades de Otimização:** Analise os dias e horários de menor volume para alocar atividades de back-office, treinamentos ou pausas prolongadas, maximizando a produtividade da equipe.
    - **Ação Estratégica:** Utilize a tabela de "Escala Recomendada pela IA" abaixo para construir sua grade de horários para as próximas semanas, garantindo uma cobertura alinhada com a demanda prevista.

    Este relatório foi gerado automaticamente pela IA da plataforma WFM Pro.
    """
    return report

# ==============================================================================
# CAMADA DE APRESENTAÇÃO (UI - STREAMLIT)
# ==============================================================================

st.title("🏆 WFM Pro AI: Plataforma Analítica de Performance")

with st.container(border=True):
    st.header("1. Carregue seu Histórico de Chamadas")
    uploaded_history = st.file_uploader("Selecione o arquivo CSV", type="csv", label_visibility="collapsed")
    
    column_mapping = {}
    if uploaded_history:
        header_df = pd.read_csv(uploaded_history, sep=None, engine='python', nrows=0)
        uploaded_history.seek(0)
        st.subheader("Mapeamento de Colunas")
        map_cols = st.columns(2)
        column_mapping['data_hora_inicio'] = map_cols[0].selectbox("Coluna de Data/Hora", header_df.columns)
        column_mapping['duracao_atendimento'] = map_cols[1].selectbox("Coluna de Duração (em s)", header_df.columns)
        if st.button("📊 Processar e Analisar Dados", use_container_width=True, type="primary"):
            st.session_state.processed_data, st.session_state.info_message = process_uploaded_file(uploaded_history, column_mapping)
            st.success(st.session_state.info_message)

if 'processed_data' not in st.session_state: st.session_state.processed_data = None

if st.session_state.processed_data is not None:
    demand_df_processed = calculate_demand(st.session_state.processed_data)
    
    tab1, tab2, tab3 = st.tabs([" dashboards Visão Geral", "💡 Otimização de Escala", "🧠 Previsão com IA"])

    with tab1:
        st.header("Dashboard de Performance Histórica")
        avg_calls = demand_df_processed['chamadas_por_hora'].mean()
        avg_tma = st.session_state.processed_data['duracao_atendimento'].mean()
        kpi1, kpi2 = st.columns(2)
        kpi1.metric("Média de Chamadas/Hora", f"{avg_calls:.1f}")
        kpi2.metric("TMA Médio Geral", f"{avg_tma:.1f}s")
        
        st.subheader("Análise Interativa: Volume vs. TMA por Hora")
        hourly_agg = demand_df_processed.groupby('Hora', as_index=False).agg({'chamadas_por_hora': 'mean', 'tma_medio': 'mean'})
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=hourly_agg['Hora'], y=hourly_agg['chamadas_por_hora'], name='Volume de Chamadas'))
        fig.add_trace(go.Scatter(x=hourly_agg['Hora'], y=hourly_agg['tma_medio'], name='TMA Médio (s)', yaxis='y2', mode='lines+markers'))
        fig.update_layout(
            title_text="<b>Volume de Chamadas vs. TMA por Hora (Média Semanal)</b>",
            template="plotly_white",
            yaxis=dict(title='Volume Médio de Chamadas'),
            yaxis2=dict(title='TMA Médio (s)', overlaying='y', side='right'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Analise a correlação entre o volume de chamadas e o tempo médio de atendimento ao longo do dia.")

    with tab2:
        st.header("Otimização de Escala (Baseada no Histórico)")
        opt_cols = st.columns(2)
        meta_sl = opt_cols[0].slider("Meta de Nível de Serviço (%)", 50, 100, 90, help="O % de chamadas atendidas dentro do tempo alvo.") / 100.0
        shrinkage = opt_cols[1].slider("Fator de Shrinkage (%)", 0, 50, 25, help="O % de tempo em que um atendente não está disponível (pausas, etc.).") / 100.0
        meta_tempo = 15 # Hardcoded ou pode ser um st.number_input

        demand_df_processed['Atendentes na Escala'] = calculate_required_staff(demand_df_processed, meta_sl, meta_tempo)
        demand_df_processed['Atendentes na Escala'] = demand_df_processed['Atendentes na Escala'].apply(lambda x: 0 if x == 0 else math.ceil(x / (1 - shrinkage)))
        
        st.subheader("Mapa de Calor Interativo da Escala Otimizada")
        heatmap_data = demand_df_processed.pivot_table(index='Dia da Semana', columns='Hora', values='Atendentes na Escala').fillna(0)
        dias_ordem_pt = ['Segunda-feira', 'Terça-feira', 'Quarta-feira', 'Quinta-feira', 'Sexta-feira', 'Sábado', 'Domingo']
        heatmap_data = heatmap_data.reindex(dias_ordem_pt)
        
        hover_text_data = demand_df_processed.pivot_table(index='Dia da Semana', columns='Hora', values=['chamadas_por_hora', 'tma_medio'])
        hover_text_data = hover_text_data.reindex(dias_ordem_pt)
        
        hover_text = hover_text_data.apply(lambda row: [f"Volume: {vol:.1f}<br>TMA: {tma:.0f}s" for vol, tma in zip(row['chamadas_por_hora'], row['tma_medio'])], axis=1)

        fig_heatmap = px.imshow(
            heatmap_data,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Viridis",
            labels=dict(x="Hora do Dia", y="Dia da Semana", color="Atendentes"),
            title="<b>Necessidade de Atendentes por Dia e Hora</b>"
        )
        fig_heatmap.update_traces(hovertemplate='Dia: %{y}<br>Hora: %{x}:00<br>Atendentes: %{z}<br>%{customdata}')
        fig_heatmap.update_traces(customdata=hover_text)
        st.plotly_chart(fig_heatmap, use_container_width=True)
        st.caption("Passe o mouse sobre as células para ver o volume e o TMA esperados para cada intervalo.")
        
        excel_data_optimized = to_excel({'Escala Otimizada': heatmap_data.reset_index()})
        st.download_button("📥 Exportar Escala para Excel (.xlsx)", excel_data_optimized, "escala_otimizada.xlsx")


    with tab3:
        st.header("Geração de Escala Futura com Inteligência Artificial")
        today = date.today()
        ia_cols = st.columns(2)
        start_date = ia_cols[0].date_input("Data de Início da Previsão", today)
        end_date = ia_cols[1].date_input("Data de Fim da Previsão", today + pd.Timedelta(days=6))

        if st.button("🚀 Gerar Previsão e Escala com IA", use_container_width=True, type="primary"):
            with st.spinner("A IA está aprendendo com seus dados..."):
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
                st.caption("A área azul representa o intervalo de confiança da previsão. Quanto mais larga, maior a incerteza do modelo para aquele período.")
                
                st.subheader("Escala Detalhada Recomendada pela IA")
                display_cols = ['ds', 'Dia da Semana', 'Hora', 'chamadas_previstas', 'tma_medio', 'Escala Prevista com IA']
                final_schedule_display = final_schedule[display_cols].rename(columns={'ds': 'Data', 'tma_medio': 'TMA Previsto (s)'})
                st.dataframe(final_schedule_display, hide_index=True)
                
                excel_data_ai = to_excel({
                    'Relatorio_IA': ai_report,
                    'Escala_Prevista_IA': final_schedule_display
                })
                st.download_button("📥 Exportar Relatório e Escala da IA para Excel (.xlsx)", excel_data_ai, "previsao_escala_ia.xlsx")