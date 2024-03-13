import streamlit as st
import pandas as pd
import plotly.express as px
from lifelines import KaplanMeierFitter
from lifelines.datasets import load_dd

# Chargement des données
data = load_dd()

# Titre de l'application
st.title('Analyse de survie avec Streamlit')

# Affichage des données brutes en option
if st.checkbox('Afficher les données brutes'):
    st.subheader('Données brutes')
    st.write(data)

# Sélection du type de visualisation
visualization_option = st.sidebar.radio(
    'Sélectionnez le type de visualisation',
    ('Statistiques descriptives', 'Histogrammes', 'Courbe de survie', 'Courbe de Kaplan-Meier par type de régime')
)

# Statistiques descriptives de la durée
if visualization_option == 'Statistiques descriptives':
    st.subheader('Statistiques descriptives de la durée :')
    
    # Calcul des statistiques descriptives
    descriptive_stats = data['duration'].describe()
    
    # Affichage du tableau avec une mise en forme améliorée
    st.table(descriptive_stats)

# Affichage de l'histogramme de la durée pour l'ensemble de la population
elif visualization_option == 'Histogrammes':
    st.subheader("Histogramme de la durée pour l'ensemble de la population")
    fig = px.histogram(data, x='duration', nbins=20, title="Histogramme de la durée", labels={'duration': 'Durée'})
    st.plotly_chart(fig)

    # Affichage de l'histogramme de la durée par type de régime
    st.subheader("Histogramme de la durée par type de régime")
    fig = px.histogram(data, x='duration', color='regime', nbins=20, title="Histogramme de la durée par type de régime", labels={'duration': 'Durée'})
    st.plotly_chart(fig)

# Estimation de la probabilité de survie avec Kaplan-Meier
elif visualization_option == 'Courbe de survie':
    st.subheader("Estimation de la probabilité de survie avec Kaplan-Meier")
    kmf = KaplanMeierFitter()
    T = data["duration"]
    E = data["observed"]
    kmf.fit(T, event_observed=E)

    # Affichage du tableau des proportions de survivants à l'instant t
    st.write("Tableau des proportions de survivants à l'instant t :")
    st.write(kmf.survival_function_)

    # Représentation graphique de la courbe de survie avec intervalle de confiance
    st.subheader("Courbe de survie avec intervalle de confiance")
    fig = px.area(
        x=kmf.survival_function_.index,
        y=kmf.survival_function_['KM_estimate'],
        title="Courbe de survie avec intervalle de confiance",
        labels={"x": "Durée", "y": "Probabilité de survie"},
        template="plotly",
    )
    ci = kmf.confidence_interval_
    fig.add_scatter(x=ci.index, y=ci['KM_estimate_lower_0.95'], mode='lines', fill='tonexty', line=dict(color='rgba(0,0,255,0.3)'), name='Intervalle de confiance (95%)')
    fig.add_scatter(x=ci.index, y=ci['KM_estimate_upper_0.95'], mode='lines', line=dict(color='rgba(0,0,255,0)'), showlegend=False)
    st.plotly_chart(fig)

# Représentation de la courbe de Kaplan-Meier pour chacun des deux groupes
elif visualization_option == 'Courbe de Kaplan-Meier par type de régime':
    st.subheader("Courbe de Kaplan-Meier par type de régime")
    kmf = KaplanMeierFitter()
    fig = px.area(title="Courbe de Kaplan-Meier par type de régime", labels={"x": "Durée", "y": "Probabilité de survie"}, template="plotly")
    for regime in data['regime'].unique():
        kmf.fit(data[data['regime'] == regime]['duration'], event_observed=data[data['regime'] == regime]['observed'], label=regime)
        fig.add_scatter(x=kmf.survival_function_.index, y=kmf.survival_function_[regime], mode='lines', name=f'{regime} - Kaplan-Meier')
        ci = kmf.confidence_interval_
        fig.add_scatter(x=ci.index, y=ci[f'{regime}_lower_0.95'], mode='lines', fill='tonexty', line=dict(color='rgba(0,0,255,0.3)'), name=f'Intervalle de confiance (95%) - {regime}')
        fig.add_scatter(x=ci.index, y=ci[f'{regime}_upper_0.95'], mode='lines', line=dict(color='rgba(0,0,255,0)'), showlegend=False)
    st.plotly_chart(fig)

