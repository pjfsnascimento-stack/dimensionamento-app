import streamlit as st
import pandas as pd
from prophet import Prophet
from src.calculations import calculate_required_staff
import math

def prepare_data_for_prophet(_df):
    df_prophet = _df.set_index('data_hora_inicio').resample('H').size().reset_index(name='y')
    return df_prophet.rename(columns={'data_hora_inicio': 'ds'})

def train_prophet_model(_df_prophet):
    model = Prophet(weekly_seasonality=True, daily_seasonality=True, yearly_seasonality=False)
    model.fit(_df_prophet)
    return model

def generate_forecast(model, start_date, end_date):
    future_periods = (end_date - start_date).days * 24 + 23
    future = model.make_future_dataframe(periods=future_periods, freq='H')
    forecast = model.predict(future)
    return forecast[(forecast['ds'].dt.date >= start_date) & (forecast['ds'].dt.date <= end_date)]

def generate_ai_staffing_schedule(_forecast, _historical_demand, meta_sl, meta_tempo, shrinkage):
    final_schedule = _forecast[['ds', 'yhat']].rename(columns={'yhat': 'chamadas_previstas'})
    final_schedule['chamadas_previstas'] = final_schedule['chamadas_previstas'].apply(lambda x: max(0, x))
    day_name_map_pt = {'Monday': 'Segunda-feira', 'Tuesday': 'Terça-feira', 'Wednesday': 'Quarta-feira','Thursday': 'Quinta-feira', 'Friday': 'Sexta-feira', 'Saturday': 'Sábado', 'Sunday': 'Domingo'}
    final_schedule['Dia da Semana'] = final_schedule['ds'].dt.day_name().map(day_name_map_pt)
    final_schedule['Hora'] = final_schedule['ds'].dt.hour
    tma_map = _historical_demand.groupby(['Dia da Semana', 'Hora'])['tma_medio'].mean().reset_index()
    final_schedule = pd.merge(final_schedule, tma_map, on=['Dia da Semana', 'Hora'], how='left').fillna(_historical_demand['tma_medio'].mean())
    final_schedule['Atendentes Necessários'] = calculate_required_staff(final_schedule.rename(columns={'chamadas_previstas': 'chamadas_por_hora'}), meta_sl, meta_tempo)
    final_schedule['Escala Prevista com IA'] = final_schedule['Atendentes Necessários'].apply(lambda x: 0 if x == 0 else math.ceil(x / (1 - shrinkage)))
    return final_schedule

def generate_ai_report(_forecast_df, _historical_demand):
    return "Relatório da IA gerado com sucesso."