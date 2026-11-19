import streamlit as st
import pandas as pd
import numpy as np

st.title('Portfolio Optimization')
st.write('La mia prima app Streamlit')

with st.sidebar:
    st.write('Impostazioni')
    mostra = st.toggle('Mostra grafici')
    tipo = st.radio('Tipo di grafico', ['Scatter', 'Bar'])

dati = pd.DataFrame({'Rischio': np.random.rand(20),
                     'Rendimento': np.random.rand(20)})

if mostra:
    if tipo == 'Scatter':
        st.scatter_chart(dati, x='Rischio', y='Rendimento')
    else:
        st.bar_chart(dati['Rendimento'])

if st.button('Cliccami'):
    st.write('Hai cliccato il pulsante!')

