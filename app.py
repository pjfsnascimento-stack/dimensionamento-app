# ==============================================================================
# PLATAFORMA DE WORKFORCE MANAGEMENT (WFM) - VERSÃO 16.1 (ESTÁVEL)
# Corrige o NameError crítico relacionado à falta da importação do módulo 'math'.
# ==============================================================================
import streamlit as st
import pandas as pd
from datetime import date, timedelta
import numpy as np
import math # <-- CORREÇÃO: Importa a biblioteca 'math' que estava faltando.

# Importa as funções dos módulos da pasta src
from src.utils import to_excel, process_uploaded_file
from src.calculations import calculate_demand, calculate_required_staff
from src.ai_model import prepare_data_for_prophet, train_prophet_model, generate_forecast, generate_ai_staffing_schedule, generate_ai_report
from src.financial import generate_cost_ai_report
from src.plotting import plot_volume_vs_tma, plot_agent_performance, plot_demand_by_client, plot_staffing_heatmap, plot_pareto_chart
from prophet.plot import plot_components_plotly

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="WFM Pro AI - Plataforma de BI", page_icon="👑", layout="wide")

st.title("👑 WFM Pro AI: Plataforma de Business Intelligence")

if 'processed_data' not in st.session_state: st.session_state.processed_data = None

with st.container(border=True):
    st.header("1. Comece sua Análise")
    demo_mode = st.toggle("Ativar Modo Demo", help="Carregue um conjunto de dados de exemplo para explorar a plataforma.")
    
    if demo_mode:
        st.session_state.processed_data = pd.DataFrame({'data_hora_inicio': pd.to_datetime(pd.date_range(end=pd.Timestamp.now(), periods=1000, freq='H')),'duracao_atendimento': [180 + math.sin(i/24)*60 + np.random.randint(-30, 30) for i in range(1000)],'Condomínio': [f"Condomínio {chr(65 + np.random.randint(0, 5))}" for _ in range(1000)],'Atendente': [f"Atendente {np.random.randint(1, 11)}" for _ in range(1000)]})
        day_name_map_pt = {'Monday': 'Segunda-feira', 'Tuesday': 'Terça-feira', 'Wednesday': 'Quarta-feira','Thursday': 'Quinta-feira', 'Friday': 'Sexta-feira', 'Saturday': 'Sábado', 'Sunday': 'Domingo'}; st.session_state.processed_data['Dia da Semana'] = st.session_state.processed_data['data_hora_inicio'].dt.day_name().map(day_name_map_pt); st.session_state.processed_data['Hora'] = st.session_state.processed_data['data_hora_inicio'].dt.hour; st.info("Modo Demo ativado.")
    else:
        uploaded_history = st.file_uploader("Ou carregue seu Histórico de Chamadas (.csv)", type="csv", label_visibility="collapsed")
        if uploaded_history:
            header_df = pd.read_csv(uploaded_history, sep=None, engine='python', nrows=0); uploaded_history.seek(0)
            st.subheader("Mapeamento de Colunas"); map_cols = st.columns(4)
            column_mapping = {'data_hora_inicio': map_cols[0].selectbox("Data/Hora (*)", header_df.columns),'duracao_atendimento': map_cols[1].selectbox("Duração (s) (*)", header_df.columns), 'Condomínio': map_cols[2].selectbox("Fila/Cliente", [None] + list(header_df.columns)),'Atendente': map_cols[3].selectbox("Atendente", [None] + list(header_df.columns)),}
            if st.button("📊 Processar e Analisar Dados", use_container_width=True, type="primary"):
                try:
                    st.session_state.processed_data, st.session_state.info_message = process_uploaded_file(uploaded_history, column_mapping)
                    st.success(st.session_state.info_message)
                except Exception as e:
                    st.error(f"Erro ao processar o arquivo: {e}")
                    st.session_state.processed_data = None


if st.session_state.processed_data is not None:
    df = st.session_state.processed_data
    
    st.header("2. Filtros Globais da Análise"); filter_cols = st.columns(4)
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
        st.subheader("Dashboard Executivo Personalizável"); dashboard_components = st.multiselect("Selecione os componentes:", ["KPIs Principais", "Volume vs. TMA por Hora", "Performance por Atendente", "Demanda por Fila/Cliente"], default=["KPIs Principais", "Volume vs. TMA por Hora"])
        if "KPIs Principais" in dashboard_components: st.markdown("---"); kpi1, kpi2, kpi3 = st.columns(3); kpi1.metric("Total de Chamadas", f"{len(df_filtered):,}"); kpi2.metric("Chamadas/Hora", f"{demand_df_filtered['chamadas_por_hora'].mean():.1f}"); kpi3.metric("TMA Geral", f"{df_filtered['duracao_atendimento'].mean():.1f}s")
        if "Volume vs. TMA por Hora" in dashboard_components: st.markdown("---"); st.plotly_chart(plot_volume_vs_tma(demand_df_filtered.groupby('Hora', as_index=False).agg(chamadas_por_hora=('chamadas_por_hora', 'mean'), tma_medio=('tma_medio', 'mean'))), use_container_width=True)
        if "Performance por Atendente" in dashboard_components and 'Atendente' in df.columns: st.markdown("---"); st.plotly_chart(plot_agent_performance(df_filtered.groupby('Atendente').agg(chamadas_atendidas=('data_hora_inicio', 'count'), tma_medio=('duracao_atendimento', 'mean')).sort_values('chamadas_atendidas', ascending=False).reset_index()), use_container_width=True)
        if "Demanda por Fila/Cliente" in dashboard_components and 'Condomínio' in df.columns: st.markdown("---"); st.plotly_chart(plot_demand_by_client(df_filtered.groupby('Condomínio').agg(chamadas_atendidas=('data_hora_inicio', 'count'), tma_medio=('duracao_atendimento', 'mean')).sort_values('chamadas_atendidas', ascending=False).reset_index()), use_container_width=True)

    with tab2:
        st.header("Otimização de Escala (Baseada no Histórico)"); opt_cols = st.columns(2)
        meta_sl = opt_cols[0].slider("Meta de Nível de Serviço (%)", 50, 100, 90, help="O % de chamadas atendidas dentro do tempo alvo.") / 100.0
        shrinkage = opt_cols[1].slider("Fator de Shrinkage (%)", 0, 50, 25, help="O % de tempo não disponível do atendente.") / 100.0
        meta_tempo = 15
        demand_df_filtered['Atendentes na Escala'] = calculate_required_staff(demand_df_filtered, meta_sl, meta_tempo)
        demand_df_filtered['Atendentes na Escala'] = demand_df_filtered['Atendentes na Escala'].apply(lambda x: 0 if x == 0 else math.ceil(x / (1 - shrinkage)))
        st.subheader("Mapa de Calor da Escala Otimizada"); heatmap_data = demand_df_filtered.pivot_table(index='Dia da Semana', columns='Hora', values='Atendentes na Escala').fillna(0); dias_ordem_pt = ['Segunda-feira', 'Terça-feira', 'Quarta-feira', 'Quinta-feira', 'Sexta-feira', 'Sábado', 'Domingo']; heatmap_data = heatmap_data.reindex(dias_ordem_pt).fillna(0)
        st.plotly_chart(plot_staffing_heatmap(heatmap_data), use_container_width=True)
        st.download_button("📥 Exportar para Excel", to_excel({'Escala Otimizada': heatmap_data.reset_index()}), "escala_otimizada.xlsx")

    with tab3:
        st.header("Geração de Escala Futura com Inteligência Artificial"); today = date.today(); ia_cols = st.columns(2)
        start_date = ia_cols[0].date_input("Data de Início da Previsão", today); end_date = ia_cols[1].date_input("Data de Fim da Previsão", today + timedelta(days=6))
        if st.button("🚀 Gerar Previsão com IA", use_container_width=True, type="primary"):
            with st.spinner("A IA está aprendendo e gerando a previsão..."):
                df_prophet = prepare_data_for_prophet(st.session_state.processed_data)
                model = train_prophet_model(df_prophet)
                forecast = generate_forecast(model, start_date, end_date)
                final_schedule = generate_ai_staffing_schedule(forecast, demand_df_filtered, meta_sl, meta_tempo, shrinkage)
                ai_report = generate_ai_report(forecast, demand_df_filtered)
                st.subheader("Relatório Executivo da IA"); st.markdown(ai_report)
                st.subheader("Análise dos Componentes da Previsão"); st.plotly_chart(plot_components_plotly(model, forecast), use_container_width=True)
                st.subheader("Escala Detalhada Recomendada pela IA"); st.dataframe(final_schedule[['ds', 'Dia da Semana', 'Hora', 'chamadas_previstas', 'tma_medio', 'Escala Prevista com IA']].rename(columns={'ds': 'Data'}), hide_index=True)
                st.download_button("📥 Exportar Relatório e Escala da IA", to_excel({'Relatorio_IA': ai_report, 'Escala_Prevista_IA': final_schedule}), "previsao_escala_ia.xlsx")
    
    with tab4:
        st.header("Análise de Custos e Rentabilidade")
        with st.container(border=True):
            st.subheader("Parâmetros Financeiros"); cost_cols = st.columns(2)
            total_payroll = cost_cols[0].number_input("Custo Total da Folha no Mês (R$)", min_value=0.0, value=50000.0, step=1000.0)
            total_agents = cost_cols[1].number_input("Total de Colaboradores na Folha", min_value=1, value=10, step=1)
        
        if total_payroll > 0 and 'Condomínio' in df.columns and 'Atendente' in df.columns:
            horas_produtivas_mes = 176
            cost_per_hour = total_payroll / (total_agents * horas_produtivas_mes) if total_agents > 0 else 0
            cost_per_second = cost_per_hour / 3600
            df_filtered['custo_atendimento'] = df_filtered['duracao_atendimento'] * cost_per_second
            
            cost_by_condo = df_filtered.groupby('Condomínio').agg(Custo_Total_Rs=('custo_atendimento', 'sum')).sort_values('Custo_Total_Rs', ascending=False).reset_index().rename(columns={'Custo_Total_Rs': 'Custo Total (R$)'})
            cost_by_condo['Percentual do Custo (%)'] = (cost_by_condo['Custo Total (R$)'] / cost_by_condo['Custo Total (R$)'].sum()) * 100
            cost_by_agent = df_filtered.groupby('Atendente').agg(Valor_Atendido_Rs=('custo_atendimento', 'sum')).sort_values('Valor_Atendido_Rs', ascending=False).reset_index().rename(columns={'Valor_Atendido_Rs': 'Valor Atendido (R$)'})

            st.markdown("---")
            kpi_cost1, kpi_cost2 = st.columns(2); kpi_cost1.metric("Custo Efetivo por Hora/Atendente", f"R$ {cost_per_hour:,.2f}"); kpi_cost2.metric("Custo Médio por Atendimento", f"R$ {df_filtered['custo_atendimento'].mean():,.2f}")
            
            ai_cost_report = generate_cost_ai_report(cost_by_condo, cost_by_agent, df_filtered['custo_atendimento'].mean())
            st.subheader("Relatório Executivo da IA Financeira"); st.markdown(ai_cost_report)

            st.subheader("Análise de Custo por Cliente (Pareto)")
            cost_by_condo['Cum_Percentual'] = cost_by_condo['Percentual do Custo (%)'].cumsum()
            st.plotly_chart(plot_pareto_chart(cost_by_condo), use_container_width=True)
            
            st.download_button("📥 Exportar Análise de Custos", to_excel({'Relatorio_IA_Custos': ai_cost_report, 'Custo_por_Cliente': cost_by_condo, 'Valor_por_Atendente': cost_by_agent}), "analise_de_custos.xlsx")
        else:
            st.warning("Para a análise de custos, mapeie as colunas 'Fila/Cliente' e 'Atendente'.")
