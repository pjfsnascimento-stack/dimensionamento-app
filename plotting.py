import plotly.express as px
import plotly.graph_objects as go

def style_figure(fig, title):
    fig.update_layout(
        title_text=f"<b>{title}</b>", template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def plot_volume_vs_tma(hourly_agg):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=hourly_agg['Hora'], y=hourly_agg['chamadas_por_hora'], name='Volume', text=hourly_agg['chamadas_por_hora'].round(1)))
    fig.add_trace(go.Scatter(x=hourly_agg['Hora'], y=hourly_agg['tma_medio'], name='TMA (s)', yaxis='y2'))
    fig.update_layout(yaxis2=dict(overlaying='y', side='right'))
    return style_figure(fig, "Volume vs. TMA por Hora")

def plot_agent_performance(agent_perf):
    fig_agent = px.bar(agent_perf.head(20), x='Atendente', y='chamadas_atendidas', color='tma_medio', text_auto='.2s', color_continuous_scale='RdYlGn_r')
    return style_figure(fig_agent, "Top 20 Atendentes por Volume vs. TMA")

def plot_demand_by_client(condo_perf):
    fig_condo = px.treemap(condo_perf, path=[px.Constant("Todos"), 'Condomínio'], values='chamadas_atendidas', color='tma_medio', color_continuous_scale='Blues', custom_data=['tma_medio'])
    fig_condo.update_traces(textinfo="label+value", hovertemplate='<b>%{label}</b><br>Chamadas: %{value}<br>TMA: %{customdata[0]:.0f}s')
    return style_figure(fig_condo, "Distribuição de Demanda por Cliente")

def plot_staffing_heatmap(heatmap_data):
    fig_heatmap = px.imshow(heatmap_data, text_auto=True, aspect="auto", color_continuous_scale="Viridis")
    return style_figure(fig_heatmap, "Necessidade de Atendentes por Dia e Hora")

def plot_pareto_chart(cost_by_condo):
    fig_pareto = go.Figure()
    fig_pareto.add_trace(go.Bar(x=cost_by_condo['Condomínio'], y=cost_by_condo['Percentual do Custo (%)'], name='Custo (%)'))
    fig_pareto.add_trace(go.Scatter(x=cost_by_condo['Condomínio'], y=cost_by_condo['Cum_Percentual'], name='Acumulado (%)', yaxis='y2'))
    fig_pareto.update_layout(yaxis2=dict(overlaying='y', side='right'))
    return style_figure(fig_pareto, "Concentração de Custo por Cliente (Pareto)")