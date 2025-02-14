import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Charger le modèle
model = pickle.load(open('model.sav', 'rb'))
expected_features = model.feature_names_in_

# Configuration de la sidebar
st.sidebar.header("ℹ️ Informations")
st.sidebar.write("""
# 🏥 Prédiction des crises cardiaques  
Cet outil évalue votre risque de crise cardiaque en fonction de votre mode de vie et antécédents médicaux.

💡 **Auteur :** Parfait Tanoh N'goran  
📌 **Données :** heart_attack_youth_vs_adult_nigeria.csv  
""")


# 🎨 Mise en page principale
st.title("❤️ Prédiction des crises cardiaques au Nigeria")

# 📌 Collecte des données utilisateur
st.subheader("📝 Remplissez les informations suivantes :")

user_inputs = {
    'Age_Group': st.selectbox("👶👴 Groupe d'âge", [0, 1], format_func=lambda x: "Jeune" if x == 0 else "Adulte"),
    'Gender': st.selectbox("🚻 Sexe", [0, 1], format_func=lambda x: "Femme" if x == 0 else "Homme"),
    'Smoking_Status': st.selectbox("🚬 Fumeur ?", [0, 1], format_func=lambda x: "Oui" if x == 0 else "Non"),
    'Exercise_Frequency': st.slider("🏃‍♂️ Fréquence d'exercice (0: Jamais - 2: Quotidien)", 0, 3, 1),
    'Hypertension': st.radio("⚠️ Hypertension ?", [0, 1], format_func=lambda x: "Non" if x == 0 else "Oui"),
    'BMI' : st.text_input("⚖️ (BMI) indique le rapport entre votre poids et votre taille", '25.5'),
    'Diabetes': st.radio("🩸 Diabète ?", [0, 1], format_func=lambda x: "Non" if x == 0 else "Oui"),
    'Stress_Level': st.slider("😟 Niveau de stress (0: Bas - 2: Élevé)", 0, 2, 1),
    'Hospitalized': st.radio("🏥 Hospitalisation récente ?", [0, 1], format_func=lambda x: "Non" if x == 0 else "Oui"),
    'Cholesterol_Level': st.radio("🫀 Niveau de cholestérol", [0, 1, 2], format_func=lambda x: ["Limite", "Élevé", "Normal"][x])
}

# 📊 Historique des prédictions
if "history" not in st.session_state:
    st.session_state.history = []

def predict():
    # Convertir en DataFrame et ajuster les colonnes
    input_df = pd.DataFrame([user_inputs]).reindex(columns=expected_features, fill_value=0)

    # Prédiction et probabilité
    prediction = model.predict(input_df)[0]
    probas = model.predict_proba(input_df)[0]

    # Sauvegarder dans l'historique
    st.session_state.history.append({'Inputs': user_inputs, 'Proba': probas, 'Prédiction': prediction})

    # 🎯 Affichage des résultats avec un graphique circulaire
    st.subheader("📊 Résultat de votre évaluation :")
    labels = ["Faible", "Modéré", "Élevé"]
    colors = ['#2ECC71', '#F1C40F', '#E74C3C']
    
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(probas, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90, wedgeprops={"edgecolor": "black"})
    st.pyplot(fig)

    # 🔴 Affichage du message selon le risque
    if prediction == 0:
        st.success(f"✅ La probabilité que vous ayez une crise cardiaque est **faible ({probas[0] * 100:.2f}%)**.")
    elif prediction == 1:
        st.warning(f"⚠️ La probabilité est **modérée ({probas[1] * 100:.2f}%)**, consultez un médecin.")
    else:
        st.error(f"🚨 **Risque élevé ({probas[2] * 100:.2f}%)**, consultez un médecin immédiatement !")

# 🎯 Bouton de prédiction
if st.button('🔍 Prédire'):
    predict()

# 📌 Affichage de l'historique
if st.sidebar.checkbox("🕘 Voir l'historique des prédictions"):
    st.sidebar.subheader("📜 Historique")
    for i, item in enumerate(reversed(st.session_state.history)):
        st.sidebar.write(f"**Prédiction {i+1}** → {['Faible', 'Modéré', 'Élevé'][item['Prédiction']]}")
