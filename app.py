# ==============================================================================
# PLATAFORMA DE WORKFORCE MANAGEMENT (WFM) - VERSÃO 13.0 (EDIÇÃO ANALÍTICA AVANÇADA)
# Implementa IA confiável, dashboards de BI, relatórios textuais e exportação universal.
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
from prophet.plot import plot_components_plotly
from datetime import date, timedelta
from openpyxl import Workbook
import numpy as np

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="WFM Pro AI - Plataforma de BI",
    page_icon="👑",
    layout="wide"
)

# ==============================================================================
# MÓDULO DE HELPERS E EXPORTAÇÃO
# ==============================================================================

@st.cache_data
def to_excel(data_dict: dict) -> bytes:
    """Converte um dicionário de DataFrames/texto para um arquivo Excel multi-abas."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheet_name, data in data_dict.items():
            if isinstance(data, pd.DataFrame):
                data.to_excel(writer, sheet_name=sheet_name, index=False)
            else:
                df_text = pd.DataFrame([{'Relatório': data}])
                df_text.to_excel(writer, sheet_name=sheet_name, index=False)
                worksheet = writer.sheets[sheet_name]
                worksheet.column_dimensions['A'].width = 120
                for cell in worksheet['A']:
                    cell.alignment = Alignment(wrap_text=True, vertical='top')
    return output.getvalue()

def style_figure(fig, title):
    """Aplica um estilo profissional e consistente aos gráficos Plotly."""
    fig.update_layout(
        title_text=f"<b>{title}</b>",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(family="sans-serif", size=12, color="black")
    )
    return fig

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

def calculate_required_staff(_demand_df, meta_sl, meta_tempo):
    @st.cache_data # Cache na função interna garante que ela recalcule se os parâmetros mudarem
    def _get_required_staff_binary_search(chamadas, tma, meta_sl, meta_tempo):
        if chamadas <= 0 or tma <= 0: return 0
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
    return [_get_required_staff_binary_search(row['chamadas_por_hora'], row['tma_medio'], meta_sl, meta_tempo) for _, row in _demand_df.iterrows()]

# Funções de IA sem cache para garantir dinamismo
def prepare_data_for_prophet(_df):
    df_prophet = _df.set_index('data_hora_inicio').resample('H').size().reset_index(name='y')
    df_prophet.rename(columns={'data_hora_inicio': 'ds'}, inplace=True)
    return df_prophet

def train_prophet_model(_df_prophet):
    model = Prophet(weekly_seasonality=True, daily_seasonality=True, yearly_seasonality=True)
    model.fit(_df_prophet)
    return model

def generate_forecast(model, start_date, end_date):
    future_periods = (end_date - start_date).days * 24 + 23
    future = model.make_future_dataframe(periods=future_periods, freq='H')
    forecast = model.predict(future)
    forecast['ds'] = pd.to_datetime(forecast['ds'])
    mask = (forecast['ds'].dt.date >= start_date) & (forecast['ds'].dt.date <= end_date)
    return forecast[mask]

def generate_ai_staffing_schedule(_forecast, _historical_demand, meta_sl, meta_tempo, shrinkage):
    final_schedule = _forecast[['ds', 'yhat']].rename(columns={'yhat': 'chamadas_previstas'})
    final_schedule['chamadas_previstas'] = final_schedule['chamadas_previstas'].apply(lambda x: max(0, x))
    day_name_map_pt = {'Monday': 'Segunda-feira', 'Tuesday': 'Terça-feira', 'Wednesday': 'Quarta-feira','Thursday': 'Quinta-feira', 'Friday': 'Sexta-feira', 'Saturday': 'Sábado', 'Sunday': 'Domingo'}
    final_schedule['Dia da Semana'] = final_schedule['ds'].dt.day_name().map(day_name_map_pt)
    final_schedule['Hora'] = final_schedule['ds'].dt.hour
    tma_map = _historical_demand.groupby(['Dia da Semana', 'Hora'])['tma_medio'].mean().reset_index()
    final_schedule = pd.merge(final_schedule, tma_map, on=['Dia da Semana', 'Hora'], how='left')
    final_schedule['tma_medio'].fillna(_historical_demand['tma_medio'].mean(), inplace=True)
    final_schedule['Atendentes Necessários'] = calculate_required_staff(final_schedule.rename(columns={'chamadas_previstas': 'chamadas_por_hora'}), meta_sl, meta_tempo)
    final_schedule['Escala Prevista com IA'] = final_schedule['Atendentes Necessários'].apply(lambda x: 0 if x == 0 else math.ceil(x / (1 - shrinkage)))
    return final_schedule

def generate_ai_report(_forecast_df, _historical_demand):
    forecast_df = _forecast_df.copy()
    forecast_df.rename(columns={'yhat': 'chamadas_previstas'}, inplace=True)
    day_name_map_pt = {'Monday': 'Segunda-feira', 'Tuesday': 'Terça-feira', 'Wednesday': 'Quarta-feira','Thursday': 'Quinta-feira', 'Friday': 'Sexta-feira', 'Saturday': 'Sábado', 'Sunday': 'Domingo'}
    forecast_df['Dia da Semana'] = forecast_df['ds'].dt.day_name().map(day_name_map_pt)
    forecast_df['Hora'] = forecast_df['ds'].dt.hour
    avg_hist_calls = _historical_demand['chamadas_por_hora'].mean()
    avg_forecast_calls = forecast_df['chamadas_previstas'].mean()
    trend_pct = ((avg_forecast_calls / avg_hist_calls) - 1) * 100 if avg_hist_calls > 0 else 0
    trend_text = f"um crescimento de **{trend_pct:.1f}%**" if trend_pct > 0 else f"um declínio de **{abs(trend_pct):.1f}%**" if trend_pct < 0 else "uma **estabilidade**"
    daily_summary = forecast_df.groupby('Dia da Semana')['chamadas_previstas'].sum().sort_values(ascending=False)
    peak_day = daily_summary.index[0]
    hourly_summary = forecast_df.groupby('Hora')['chamadas_previstas'].mean().sort_values(ascending=False)
    peak_hour = hourly_summary.index[0]
    report = f"""
    ### Relatório Executivo da Previsão de IA
    **1. Análise de Tendência Geral:** A análise projeta **{trend_text}** no volume médio de chamadas por hora em comparação com o período histórico.
    **2. Padrões de Demanda:** O dia de maior volume previsto é **{peak_day}**, com o pico de demanda ocorrendo por volta das **{peak_hour}:00**.
    **3. Ações Recomendadas:**
    - **Alocação Estratégica:** Direcione seus recursos para o período de pico e utilize horários de baixo volume para tarefas de back-office.
    - **Planejamento de Contratação:** A tendência de **{'crescimento' if trend_pct > 0 else 'declínio'}** deve informar suas decisões de contratação ou remanejamento de pessoal a médio prazo.
    """
    return report

# ==============================================================================
# CAMADA DE APRESENTAÇÃO (UI - STREAMLIT)
# ==============================================================================

st.title("👑 WFM Pro AI: Plataforma de Business Intelligence")

if 'processed_data' not in st.session_state: st.session_state.processed_data = None

with st.container(border=True):
    st.header("1. Comece sua Análise")
    demo_mode = st.toggle("Ativar Modo Demo", help="Carregue um conjunto de dados de exemplo para explorar a plataforma.")
    
    if demo_mode:
        st.session_state.processed_data = pd.DataFrame({
            'data_hora_inicio': pd.to_datetime(pd.date_range(end=pd.Timestamp.now(), periods=1000, freq='H')),
            'duracao_atendimento': [180 + math.sin(i/24)*60 + np.random.randint(-30, 30) for i in range(1000)],
            'Condomínio': [f"Condomínio {chr(65 + np.random.randint(0, 5))}" for _ in range(1000)],
            'Atendente': [f"Atendente {np.random.randint(1, 11)}" for _ in range(1000)]
        })
        day_name_map_pt = {'Monday': 'Segunda-feira', 'Tuesday': 'Terça-feira', 'Wednesday': 'Quarta-feira','Thursday': 'Quinta-feira', 'Friday': 'Sexta-feira', 'Saturday': 'Sábado', 'Sunday': 'Domingo'}
        st.session_state.processed_data['Dia da Semana'] = st.session_state.processed_data['data_hora_inicio'].dt.day_name().map(day_name_map_pt)
        st.session_state.processed_data['Hora'] = st.session_state.processed_data['data_hora_inicio'].dt.hour
        st.info("Modo Demo ativado. Um conjunto de dados de exemplo foi carregado.")
    else:
        uploaded_history = st.file_uploader("Ou carregue seu Histórico de Chamadas (.csv)", type="csv", label_visibility="collapsed")
        if uploaded_history:
            header_df = pd.read_csv(uploaded_history, sep=None, engine='python', nrows=0)
            uploaded_history.seek(0)
            
            st.subheader("Mapeamento de Colunas")
            map_cols = st.columns(4)
            column_mapping = {
                'data_hora_inicio': map_cols[0].selectbox("Data/Hora (*)", header_df.columns),
                'duracao_atendimento': map_cols[1].selectbox("Duração (s) (*)", header_df.columns),
                'Condomínio': map_cols[2].selectbox("Fila/Cliente", [None] + list(header_df.columns)),
                'Atendente': map_cols[3].selectbox("Atendente", [None] + list(header_df.columns)),
            }
            if st.button("📊 Processar e Analisar Dados", use_container_width=True, type="primary"):
                st.session_state.processed_data, st.session_state.info_message = process_uploaded_file(uploaded_history, column_mapping)
                st.success(st.session_state.info_message)

if st.session_state.processed_data is not None:
    df = st.session_state.processed_data
    
    st.header("2. Filtros Globais da Análise")
    filter_cols = st.columns(4)
    selected_days = filter_cols[0].multiselect("Dias da Semana", df['Dia da Semana'].unique(), default=df['Dia da Semana'].unique())
    time_range = filter_cols[1].slider("Intervalo de Horas", 0, 23, (0, 23))
    selected_condos = filter_cols[2].multiselect("Fila/Cliente", df['Condomínio'].unique(), default=df['Condomínio'].unique()) if 'Condomínio' in df.columns else []
    selected_agents = filter_cols[3].multiselect("Atendente", df['Atendente'].unique(), default=df['Atendente'].unique()) if 'Atendente' in df.columns else []

    df_filtered = df[(df['Dia da Semana'].isin(selected_days)) & (df['Hora'] >= time_range[0]) & (df['Hora'] <= time_range[1]) & (df['Condomínio'].isin(selected_condos) if 'Condomínio' in df.columns and selected_condos else True) & (df['Atendente'].isin(selected_agents) if 'Atendente' in df.columns and selected_agents else True)]
    demand_df_filtered = calculate_demand(df_filtered)

    st.header("3. Explore os Dados")
    tab1, tab2, tab3 = st.tabs([" B.I. Meu Dashboard", "💡 Otimização de Escala", "🧠 Previsão com IA"])

    with tab1:
        st.subheader("Dashboard Executivo Personalizável")
        dashboard_components = st.multiselect("Selecione os componentes para exibir:", ["KPIs Principais", "Volume vs. TMA por Hora", "Performance por Atendente", "Demanda por Fila/Cliente"], default=["KPIs Principais", "Volume vs. TMA por Hora"])
        
        if "KPIs Principais" in dashboard_components:
            st.markdown("---")
            kpi1, kpi2, kpi3 = st.columns(3)
            kpi1.metric("Total de Chamadas Analisadas", f"{len(df_filtered):,}")
            kpi2.metric("Média de Chamadas/Hora", f"{demand_df_filtered['chamadas_por_hora'].mean():.1f}")
            kpi3.metric("TMA Médio Geral", f"{df_filtered['duracao_atendimento'].mean():.1f}s")
        
        if "Volume vs. TMA por Hora" in dashboard_components:
            st.markdown("---")
            st.write("#### Análise Interativa: Volume vs. TMA por Hora")
            hourly_agg = demand_df_filtered.groupby('Hora', as_index=False).agg({'chamadas_por_hora': 'mean', 'tma_medio': 'mean'})
            fig = go.Figure()
            fig.add_trace(go.Bar(x=hourly_agg['Hora'], y=hourly_agg['chamadas_por_hora'], name='Volume de Chamadas', yaxis='y1', text=hourly_agg['chamadas_por_hora'].round(1), textposition='auto'))
            fig.add_trace(go.Scatter(x=hourly_agg['Hora'], y=hourly_agg['tma_medio'], name='TMA Médio (s)', yaxis='y2', mode='lines+markers'))
            fig = style_figure(fig, "Volume de Chamadas vs. TMA por Hora (Média)")
            st.plotly_chart(fig, use_container_width=True)

        if "Performance por Atendente" in dashboard_components and 'Atendente' in df.columns:
            st.markdown("---")
            st.write("#### Performance por Atendente")
            agent_perf = df_filtered.groupby('Atendente').agg(chamadas_atendidas=('data_hora_inicio', 'count'), tma_medio=('duracao_atendimento', 'mean')).sort_values(by='chamadas_atendidas', ascending=False).reset_index()
            fig_agent = px.bar(agent_perf.head(20), x='Atendente', y='chamadas_atendidas', color='tma_medio', text_auto='.2s', title="Top 20 Atendentes por Volume vs. TMA Médio", color_continuous_scale='RdYlGn_r')
            fig_agent = style_figure(fig_agent, "Top 20 Atendentes por Volume vs. TMA Médio")
            st.plotly_chart(fig_agent, use_container_width=True)

        if "Demanda por Fila/Cliente" in dashboard_components and 'Condomínio' in df.columns:
            st.markdown("---")
            st.write("#### Demanda por Fila/Cliente")
            condo_perf = df_filtered.groupby('Condomínio').agg(chamadas_atendidas=('data_hora_inicio', 'count'), tma_medio=('duracao_atendimento', 'mean')).sort_values(by='chamadas_atendidas', ascending=False).reset_index()
            fig_condo = px.treemap(condo_perf, path=[px.Constant("Todos"), 'Condomínio'], values='chamadas_atendidas', color='tma_medio', title='Distribuição de Volume e TMA por Cliente', color_continuous_scale='Blues', custom_data=['tma_medio'])
            fig_condo.update_traces(textinfo="label+value", hovertemplate='<b>%{label}</b><br>Chamadas: %{value}<br>TMA Médio: %{customdata[0]:.0f}s')
            fig_condo = style_figure(fig_condo, "Distribuição de Volume e TMA por Cliente")
            st.plotly_chart(fig_condo, use_container_width=True)

    with tab2:
        st.header("Otimização de Escala (Baseada no Histórico)")
        opt_cols = st.columns(2)
        meta_sl = opt_cols[0].slider("Meta de Nível de Serviço (%)", 50, 100, 90, help="O % de chamadas atendidas dentro do tempo alvo.") / 100.0
        shrinkage = opt_cols[1].slider("Fator de Shrinkage (%)", 0, 50, 25, help="O % de tempo não disponível do atendente.") / 100.0
        meta_tempo = 15
        
        demand_df_filtered['Atendentes na Escala'] = calculate_required_staff(demand_df_filtered, meta_sl, meta_tempo)
        demand_df_filtered['Atendentes na Escala'] = demand_df_filtered['Atendentes na Escala'].apply(lambda x: 0 if x == 0 else math.ceil(x / (1 - shrinkage)))
        
        st.subheader("Mapa de Calor Interativo da Escala Otimizada")
        heatmap_data = demand_df_filtered.pivot_table(index='Dia da Semana', columns='Hora', values='Atendentes na Escala').fillna(0)
        dias_ordem_pt = ['Segunda-feira', 'Terça-feira', 'Quarta-feira', 'Quinta-feira', 'Sexta-feira', 'Sábado', 'Domingo']
        heatmap_data = heatmap_data.reindex(dias_ordem_pt).fillna(0)
        
        fig_heatmap = px.imshow(heatmap_data, text_auto=True, aspect="auto", color_continuous_scale="Viridis")
        fig_heatmap = style_figure(fig_heatmap, "Necessidade de Atendentes por Dia e Hora")
        st.plotly_chart(fig_heatmap, use_container_width=True)
        excel_data_optimized = to_excel({'Escala Otimizada': heatmap_data.reset_index()})
        st.download_button("📥 Exportar para Excel", excel_data_optimized, "escala_otimizada.xlsx")

    with tab3:
        st.header("Geração de Escala Futura com Inteligência Artificial")
        today = date.today()
        ia_cols = st.columns(2)
        start_date = ia_cols[0].date_input("Data de Início da Previsão", today)
        end_date = ia_cols[1].date_input("Data de Fim da Previsão", today + timedelta(days=6))
        
        if st.button("🚀 Gerar Previsão com IA", use_container_width=True, type="primary"):
            with st.spinner("A IA está aprendendo e gerando a previsão..."):
                df_prophet = prepare_data_for_prophet(st.session_state.processed_data)
                model = train_prophet_model(df_prophet)
                forecast = generate_forecast(model, start_date, end_date)
                final_schedule = generate_ai_staffing_schedule(forecast, demand_df_filtered, meta_sl, meta_tempo, shrinkage)
                ai_report = generate_ai_report(forecast, demand_df_filtered)
                
                st.subheader("Relatório Executivo da IA")
                st.markdown(ai_report)
                
                st.subheader("Análise dos Componentes da Previsão")
                st.write("Este gráfico mostra como a IA enxerga os padrões da sua operação. Use os filtros para entender a tendência de longo prazo e as sazonalidades semanais e diárias.")
                # CORREÇÃO CRÍTICA: Substitui a função inexistente por uma que funciona
                fig_components = model.plot_components(forecast, plotly=True)
                fig_components = style_figure(fig_components, "Componentes da Previsão de Demanda")
                st.plotly_chart(fig_components, use_container_width=True)

                st.subheader("Escala Detalhada Recomendada pela IA")
                heatmap_ia_data = final_schedule.pivot_table(index='Dia da Semana', columns='Hora', values='Escala Prevista com IA').fillna(0).reindex(dias_ordem_pt).fillna(0)
                fig_heatmap_ia = px.imshow(heatmap_ia_data, text_auto=True, aspect="auto", color_continuous_scale="Plasma")
                fig_heatmap_ia = style_figure(fig_heatmap_ia, "Escala Prevista pela IA por Dia e Hora")
                st.plotly_chart(fig_heatmap_ia, use_container_width=True)
                
                excel_data_ai = to_excel({'Relatorio_IA': ai_report, 'Escala_Prevista_IA': final_schedule})
                st.download_button("📥 Exportar Relatório e Escala da IA", excel_data_ai, "previsao_escala_ia.xlsx")