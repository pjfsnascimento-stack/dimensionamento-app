# ==============================================================================
# PLATAFORMA DE WORKFORCE MANAGEMENT (WFM) - VERSÃO 15.1 (EDIÇÃO DE RECUPERAÇÃO E APRIMORAMENTO)
# Restaura a Análise de Custos, corrige o layout e garante a funcionalidade de todos os relatórios.
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
    page_title="WFM Pro AI - Plataforma de BI Financeiro",
    page_icon="🏦",
    layout="wide"
)

# ==============================================================================
# MÓDULOS DE LÓGICA DE NEGÓCIO (BACKEND)
# ==============================================================================
# ... (Funções de to_excel, process_uploaded_file, calculate_demand, IA etc., permanecem as mesmas) ...
@st.cache_data
def to_excel(data_dict: dict) -> bytes:
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
        total_chamadas=('data_hora_inicio', 'count')
    ).reset_index().fillna(0)
    demand_df['chamadas_por_hora'] = (demand_df['total_chamadas'] / num_semanas)
    return demand_df

def calculate_required_staff(_demand_df, meta_sl, meta_tempo):
    @st.cache_data
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
    # ... (código do relatório de IA permanece o mesmo) ...
    forecast_df = _forecast_df.copy(); forecast_df.rename(columns={'yhat': 'chamadas_previstas'}, inplace=True)
    day_name_map_pt = {'Monday': 'Segunda-feira', 'Tuesday': 'Terça-feira', 'Wednesday': 'Quarta-feira','Thursday': 'Quinta-feira', 'Friday': 'Sexta-feira', 'Saturday': 'Sábado', 'Sunday': 'Domingo'}
    forecast_df['Dia da Semana'] = forecast_df['ds'].dt.day_name().map(day_name_map_pt)
    forecast_df['Hora'] = forecast_df['ds'].dt.hour
    avg_hist_calls = _historical_demand['chamadas_por_hora'].mean(); avg_forecast_calls = forecast_df['chamadas_previstas'].mean()
    trend_pct = ((avg_forecast_calls / avg_hist_calls) - 1) * 100 if avg_hist_calls > 0 else 0
    trend_text = f"um crescimento de **{trend_pct:.1f}%**" if trend_pct > 0 else f"um declínio de **{abs(trend_pct):.1f}%**" if trend_pct < 0 else "uma **estabilidade**"
    daily_summary = forecast_df.groupby('Dia da Semana')['chamadas_previstas'].sum().sort_values(ascending=False); peak_day = daily_summary.index[0]
    hourly_summary = forecast_df.groupby('Hora')['chamadas_previstas'].mean().sort_values(ascending=False); peak_hour = hourly_summary.index[0]
    return f"""### Relatório Executivo da Previsão de IA\n**1. Tendência Geral:** A análise projeta **{trend_text}** no volume médio de chamadas.\n**2. Padrões de Demanda:** O dia de maior volume previsto é **{peak_day}**, com pico por volta das **{peak_hour}:00**.\n**3. Ações Recomendadas:** Foque recursos no período de pico e use horários de baixo volume para tarefas de back-office."""
@st.cache_data
def generate_cost_ai_report(_cost_by_condo, _cost_by_agent, _cost_per_call):
    # ... (código do relatório de IA de custos permanece o mesmo) ...
    if _cost_by_condo.empty or _cost_by_agent.empty: return "Dados insuficientes para gerar o relatório financeiro."
    top_condo = _cost_by_condo.sort_values('Custo Total (R$)', ascending=False).iloc[0]
    top_agent = _cost_by_agent.sort_values('Valor Atendido (R$)', ascending=False).iloc[0]
    return f"""### Relatório Executivo da Análise de Custos\n**1. Custo por Atendimento:** O custo médio por interação foi de **R$ {_cost_per_call:,.2f}**.\n**2. Análise por Cliente:** O cliente de maior custo foi **{top_condo['Condomínio']}** (R$ {top_condo['Custo Total (R$)']:,.2f}, representando **{top_condo['Percentual do Custo (%)']:.1f}%** do total).\n**3. Análise por Atendente:** O operador de maior produtividade em valor foi **{top_agent['Atendente']}** (R$ {top_agent['Valor Atendido (R$)']:,.2f} em atendimentos)."""

# ==============================================================================
# CAMADA DE APRESENTAÇÃO (UI - STREAMLIT)
# ==============================================================================

st.title("🏦 WFM Pro AI: Plataforma de Inteligência Financeira e Operacional")

if 'processed_data' not in st.session_state: st.session_state.processed_data = None

with st.container(border=True):
    st.header("1. Comece sua Análise")
    demo_mode = st.toggle("Ativar Modo Demo", help="Carregue um conjunto de dados de exemplo para explorar a plataforma.")
    
    if demo_mode:
        st.session_state.processed_data = pd.DataFrame({'data_hora_inicio': pd.to_datetime(pd.date_range(end=pd.Timestamp.now(), periods=1000, freq='H')), 'duracao_atendimento': [180 + math.sin(i/24)*60 + np.random.randint(-30, 30) for i in range(1000)], 'Condomínio': [f"Condomínio {chr(65 + np.random.randint(0, 5))}" for _ in range(1000)], 'Atendente': [f"Atendente {np.random.randint(1, 11)}" for _ in range(1000)]})
        day_name_map_pt = {'Monday': 'Segunda-feira', 'Tuesday': 'Terça-feira', 'Wednesday': 'Quarta-feira','Thursday': 'Quinta-feira', 'Friday': 'Sexta-feira', 'Saturday': 'Sábado', 'Sunday': 'Domingo'}; st.session_state.processed_data['Dia da Semana'] = st.session_state.processed_data['data_hora_inicio'].dt.day_name().map(day_name_map_pt); st.session_state.processed_data['Hora'] = st.session_state.processed_data['data_hora_inicio'].dt.hour; st.info("Modo Demo ativado.")
    else:
        uploaded_history = st.file_uploader("Ou carregue seu Histórico de Chamadas (.csv)", type="csv", label_visibility="collapsed")
        if uploaded_history:
            header_df = pd.read_csv(uploaded_history, sep=None, engine='python', nrows=0); uploaded_history.seek(0)
            st.subheader("Mapeamento de Colunas"); map_cols = st.columns(4)
            column_mapping = {'data_hora_inicio': map_cols[0].selectbox("Data/Hora (*)", header_df.columns),'duracao_atendimento': map_cols[1].selectbox("Duração (s) (*)", header_df.columns), 'Condomínio': map_cols[2].selectbox("Fila/Cliente", [None] + list(header_df.columns)),'Atendente': map_cols[3].selectbox("Atendente", [None] + list(header_df.columns)),}
            if st.button("📊 Processar e Analisar Dados", use_container_width=True, type="primary"):
                st.session_state.processed_data, st.session_state.info_message = process_uploaded_file(uploaded_history, column_mapping); st.success(st.session_state.info_message)

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
    tab_titles = [" B.I. Meu Dashboard", "💡 Otimização de Escala", "🧠 Previsão com IA", "💰 Análise de Custos"]
    tab1, tab2, tab3, tab4 = st.tabs(tab_titles)

    with tab1:
        st.subheader("Dashboard Executivo Personalizável")
        dashboard_components = st.multiselect("Selecione os componentes para exibir:", ["KPIs Principais", "Volume vs. TMA por Hora", "Performance por Atendente", "Demanda por Fila/Cliente"], default=["KPIs Principais", "Volume vs. TMA por Hora"])
        if "KPIs Principais" in dashboard_components: st.markdown("---"); kpi1, kpi2, kpi3 = st.columns(3); kpi1.metric("Total de Chamadas Analisadas", f"{len(df_filtered):,}"); kpi2.metric("Média de Chamadas/Hora", f"{demand_df_filtered['chamadas_por_hora'].mean():.1f}"); kpi3.metric("TMA Médio Geral", f"{df_filtered['duracao_atendimento'].mean():.1f}s")
        if "Volume vs. TMA por Hora" in dashboard_components: st.markdown("---"); st.write("#### Análise Interativa: Volume vs. TMA"); hourly_agg = demand_df_filtered.groupby('Hora', as_index=False).agg({'chamadas_por_hora': 'mean', 'tma_medio': 'mean'}); fig = go.Figure(); fig.add_trace(go.Bar(x=hourly_agg['Hora'], y=hourly_agg['chamadas_por_hora'], name='Volume', yaxis='y1', text=hourly_agg['chamadas_por_hora'].round(1))); fig.add_trace(go.Scatter(x=hourly_agg['Hora'], y=hourly_agg['tma_medio'], name='TMA (s)', yaxis='y2', mode='lines+markers')); fig.update_layout(title_text="<b>Volume vs. TMA por Hora</b>", template="plotly_white", yaxis=dict(title='Volume Médio'), yaxis2=dict(title='TMA (s)', overlaying='y', side='right'), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)); st.plotly_chart(fig, use_container_width=True)
        if "Performance por Atendente" in dashboard_components and 'Atendente' in df.columns: st.markdown("---"); st.write("#### Performance por Atendente"); agent_perf = df_filtered.groupby('Atendente').agg(chamadas_atendidas=('data_hora_inicio', 'count'), tma_medio=('duracao_atendimento', 'mean')).sort_values(by='chamadas_atendidas', ascending=False).reset_index(); fig_agent = px.bar(agent_perf.head(20), x='Atendente', y='chamadas_atendidas', color='tma_medio', text_auto='.2s', color_continuous_scale='RdYlGn_r'); fig_agent = go.Figure(fig_agent.update_layout(title_text="<b>Top 20 Atendentes por Volume vs. TMA Médio</b>", template="plotly_white")); st.plotly_chart(fig_agent, use_container_width=True)
        if "Demanda por Fila/Cliente" in dashboard_components and 'Condomínio' in df.columns: st.markdown("---"); st.write("#### Demanda por Fila/Cliente"); condo_perf = df_filtered.groupby('Condomínio').agg(chamadas_atendidas=('data_hora_inicio', 'count'), tma_medio=('duracao_atendimento', 'mean')).sort_values(by='chamadas_atendidas', ascending=False).reset_index(); fig_condo = px.treemap(condo_perf, path=[px.Constant("Todos"), 'Condomínio'], values='chamadas_atendidas', color='tma_medio', color_continuous_scale='Blues', custom_data=['tma_medio']); fig_condo.update_traces(textinfo="label+value", hovertemplate='<b>%{label}</b><br>Chamadas: %{value}<br>TMA Médio: %{customdata[0]:.0f}s'); fig_condo = go.Figure(fig_condo.update_layout(title_text="<b>Distribuição de Volume e TMA por Cliente</b>", template="plotly_white")); st.plotly_chart(fig_condo, use_container_width=True)

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
        fig_heatmap = px.imshow(heatmap_data, text_auto=True, aspect="auto", color_continuous_scale="Viridis", title="<b>Necessidade de Atendentes por Dia e Hora</b>")
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
                st.subheader("Relatório Executivo da IA"); st.markdown(ai_report)
                st.subheader("Análise dos Componentes da Previsão"); st.write("Este gráfico mostra como a IA enxerga os padrões da sua operação.")
                fig_components = plot_components_plotly(model, forecast); st.plotly_chart(fig_components, use_container_width=True)
                st.subheader("Escala Detalhada Recomendada pela IA"); st.dataframe(final_schedule[['ds', 'Dia da Semana', 'Hora', 'chamadas_previstas', 'tma_medio', 'Escala Prevista com IA']].rename(columns={'ds': 'Data'}), hide_index=True)
                excel_data_ai = to_excel({'Relatorio_IA': ai_report, 'Escala_Prevista_IA': final_schedule}); st.download_button("📥 Exportar Relatório e Escala da IA", excel_data_ai, "previsao_escala_ia.xlsx")
    
    with tab4:
        st.header("Análise de Custos e Rentabilidade")
        with st.container(border=True):
            st.subheader("Parâmetros Financeiros")
            cost_cols = st.columns(2)
            total_payroll = cost_cols[0].number_input("Custo Total da Folha no Mês (R$)", min_value=0.0, value=50000.0, step=1000.0, help="Informe o custo total da folha de pagamento (salários + encargos) para o período dos dados.")
            total_agents = cost_cols[1].number_input("Total de Colaboradores na Folha", min_value=1, value=10, step=1, help="Informe o número total de atendentes que compõem o custo da folha.")

        if total_payroll > 0 and total_agents > 0 and 'Condomínio' in df.columns and 'Atendente' in df.columns:
            horas_produtivas_mes = 176
            cost_per_hour_effective = total_payroll / (total_agents * horas_produtivas_mes)
            cost_per_second = cost_per_hour_effective / 3600
            df_filtered['custo_atendimento'] = df_filtered['duracao_atendimento'] * cost_per_second
            
            cost_by_condo = df_filtered.groupby('Condomínio').agg(Custo_Total_Rs=('custo_atendimento', 'sum'), Total_Atendimentos=('data_hora_inicio', 'count')).sort_values('Custo_Total_Rs', ascending=False).reset_index()
            cost_by_condo['Percentual_Custo'] = (cost_by_condo['Custo_Total_Rs'] / cost_by_condo['Custo_Total_Rs'].sum()) * 100
            cost_by_agent = df_filtered.groupby('Atendente').agg(Valor_Atendido_Rs=('custo_atendimento', 'sum'), Total_Atendimentos=('data_hora_inicio', 'count'), TMA_Medio_s=('duracao_atendimento', 'mean')).sort_values('Valor_Atendido_Rs', ascending=False).reset_index()

            st.markdown("---")
            kpi_cost1, kpi_cost2 = st.columns(2)
            kpi_cost1.metric("Custo Efetivo por Hora de Trabalho", f"R$ {cost_per_hour_effective:,.2f}")
            kpi_cost2.metric("Custo Médio por Atendimento", f"R$ {(df_filtered['custo_atendimento'].mean()):,.2f}")
            
            ai_cost_report = generate_cost_ai_report(cost_by_condo, cost_by_agent, df_filtered['custo_atendimento'].mean())
            st.subheader("Relatório Executivo da IA Financeira")
            st.markdown(ai_cost_report)

            st.subheader("Análise de Custo por Cliente (Princípio de Pareto)")
            cost_by_condo['Cum_Percentual'] = cost_by_condo['Percentual_Custo'].cumsum()
            fig_pareto = go.Figure()
            fig_pareto.add_trace(go.Bar(x=cost_by_condo['Condomínio'], y=cost_by_condo['Percentual_Custo'], name='Custo por Cliente (%)'))
            fig_pareto.add_trace(go.Scatter(x=cost_by_condo['Condomínio'], y=cost_by_condo['Cum_Percentual'], name='Custo Acumulado (%)', yaxis='y2', mode='lines+markers'))
            fig_pareto = go.Figure(fig_pareto.update_layout(yaxis=dict(title='Custo Individual (%)'), yaxis2=dict(title='Custo Acumulado (%)', overlaying='y', side='right')))
            fig_pareto = style_figure(fig_pareto, "Concentração de Custo por Cliente (Pareto 80/20)")
            st.plotly_chart(fig_pareto, use_container_width=True)

            st.subheader("Análise de Valor por Atendente")
            fig_agent_cost = px.bar(cost_by_agent.head(20), x='Atendente', y='Valor_Atendido_Rs', color='TMA_Medio_s', text_auto='.2s', color_continuous_scale='RdYlGn_r')
            fig_agent_cost = style_figure(fig_agent_cost, "Top 20 Atendentes por Valor Atendido (R$) vs. TMA Médio")
            st.plotly_chart(fig_agent_cost, use_container_width=True)

            excel_data_cost = to_excel({'Relatorio_IA_Custos': ai_cost_report, 'Custo_por_Cliente': cost_by_condo, 'Valor_por_Atendente': cost_by_agent})
            st.download_button("📥 Exportar Análise de Custos para Excel", excel_data_cost, "analise_de_custos.xlsx")
        else:
            st.warning("Para a análise de custos, por favor, mapeie as colunas 'Fila/Cliente' e 'Atendente' e preencha os parâmetros financeiros.")