import streamlit as st
import numpy as np
import pickle
from tensorflow import keras

st.title("Anormal Sektör Tespiti (Autoencoder) :warning:")
st.write("Bir sektörün emisyon faktörlerini girin (kg CO2e / USD). Model alışılmadık bir profil olup olmadığını söylesin.")
model=keras.models.load_model('ghg_autoencoder.keras')
scaler=pickle.load(open('ae_scaler.pkl','rb'))
esik=pickle.load(open('ae_esik.pkl','rb'))

without_margins=st.number_input('Marjsız emisyon faktörü',0.0,50.0,0.5)
margins=st.number_input('Marj emisyon faktörü',0.0,10.0,0.05)
with_margins=st.number_input('Marjlı emisyon faktörü',0.0,50.0,0.55)

if st.button('Kontrol et'):
    x=scaler.transform(np.log1p(np.array([[without_margins,margins,with_margins]])))
    hata=float(np.mean((x-model.predict(x,verbose=0))**2))
    st.write(f'Yeniden oluşturma hatası: {hata:.4f} (eşik: {esik:.4f})')
    if hata>esik:
        st.error('Anormal profil')
    else:
        st.success('Normal profil')
