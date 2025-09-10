import streamlit as st
import pandas as pd

@st.cache_data
def generate_cost_ai_report(_cost_by_condo, _cost_by_agent, _cost_per_call):
    if _cost_by_condo.empty or _cost_by_agent.empty: return "Dados insuficientes para gerar o relatório financeiro."
    top_condo = _cost_by_condo.iloc[0]
    top_agent = _cost_by_agent.iloc[0]
    return f"""### Relatório Executivo da Análise de Custos\n**1. Custo por Atendimento:** O custo médio por interação foi de **R$ {_cost_per_call:,.2f}**.\n**2. Análise por Cliente:** O cliente de maior custo foi **{top_condo['Condomínio']}** (R$ {top_condo['Custo Total (R$)']:,.2f}, representando **{top_condo['Percentual do Custo (%)']:.1f}%** do total).\n**3. Análise por Atendente:** O operador de maior produtividade em valor foi **{top_agent['Atendente']}** (R$ {top_agent['Valor Atendido (R$)']:,.2f} em atendimentos)."""