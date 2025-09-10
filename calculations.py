import streamlit as st
import pandas as pd
from pyworkforce.queuing import ErlangC
import math

@st.cache_data
def calculate_demand(_df):
    df = _df.copy()
    num_semanas = (df['data_hora_inicio'].max() - df['data_hora_inicio'].min()).days / 7 or 1
    demand_df = df.groupby(['Dia da Semana', 'Hora']).agg(
        tma_medio=('duracao_atendimento', 'mean'),
        total_chamadas=('data_hora_inicio', 'count')
    ).reset_index().fillna(0)
    demand_df['chamadas_por_hora'] = (demand_df['total_chamadas'] / num_semanas)
    return demand_df

def calculate_required_staff(_demand_df, meta_sl, meta_tempo):
    def _get_required_staff(chamadas, tma):
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
    return [_get_required_staff(row['chamadas_por_hora'], row['tma_medio']) for _, row in _demand_df.iterrows()]