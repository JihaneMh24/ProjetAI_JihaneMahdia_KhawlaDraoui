from django.shortcuts import render
from .forms import SentimentForm
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import re
from nltk.corpus import stopwords
from collections import defaultdict

# Télécharger les stopwords si nécessaire
nltk_stopwords = set(stopwords.words('english'))

# Chemins vers les modèles
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'model.pkl')
vectorizer_path = os.path.join(BASE_DIR, 'vectorizer.pkl')
positif_words_path = os.path.join(BASE_DIR, 'mots_positifs.pkl')
negatif_words_path = os.path.join(BASE_DIR, 'mots_negatifs.pkl')

# Charger le modèle et le vectoriseur
model = pickle.load(open(model_path, 'rb'))
vectorizer = pickle.load(open(vectorizer_path, 'rb'))

# Charger les sets de mots positifs et négatifs
mots_positifs_set = set(pickle.load(open(positif_words_path, 'rb')).keys())
mots_negatifs_set = set(pickle.load(open(negatif_words_path, 'rb')).keys())

def traiter_negations(texte):
    """
    Transforme les phrases avec des négations :
    - Négation + mot positif → négatif
    - Négation + mot négatif → positif
    """
    negations = ["not", "no", "don't", "isn't", "aren't", "won't", "didn't", "doesn't", "can't", "couldn't", "never"]
    mots = texte.split()
    mots_traite = []
    skip = False

    for i in range(len(mots)):
        if skip:
            skip = False
            continue

        # Vérifier si un mot est une négation suivi d'un mot positif ou négatif
        if mots[i] in negations and i + 1 < len(mots):
            suivant = mots[i + 1]  # Le mot après la négation
            if suivant in mots_positifs_set:
                mots_traite.append("negated_positive")  # Négation + positif → négatif
            if suivant in mots_negatifs_set:
                mots_traite.append("negated_negative")  # Négation + négatif → positif
            else:
                mots_traite.append(f"{mots[i]}_{suivant}")  # Gérer comme "not_happy", etc.
            skip = True  # Ignorer le mot suivant
        else:
            mots_traite.append(mots[i])

    return ' '.join(mots_traite)

def nettoyer_texte(texte):
    """
    Nettoyer le texte en supprimant les caractères spéciaux, en gérant les négations, et en retirant les stopwords.
    """
    # Supprimer les caractères spéciaux et chiffres
    texte = re.sub(r'[^a-zA-Z\s]', '', texte)
    # Convertir en minuscules
    texte = texte.lower()
    # Traiter les négations
    texte = traiter_negations(texte)
    # Supprimer les mots vides
    mots = texte.split()
    mots = [mot for mot in mots if mot not in nltk_stopwords]
    return ' '.join(mots)

def sentiment_analysis(request):
    sentiment = None
    if request.method == 'POST':
        form = SentimentForm(request.POST)
        if form.is_valid():
            text = form.cleaned_data['text']
            # Nettoyer et traiter le texte
            cleaned_text = nettoyer_texte(text)
            # Vectoriser le texte
            text_vectorized = vectorizer.transform([cleaned_text])
            # Prédire le sentiment
            prediction = model.predict(text_vectorized)[0]
            
            # Gestion des cas "negated_positive" et "negated_negative"
            if "negated_positive" in cleaned_text:
                sentiment = "Négatif"
            elif "negated_negative" in cleaned_text:
                sentiment = "Positif"
            else:
                sentiment = "Positif" if prediction == 1 else "Négatif"
    else:
        form = SentimentForm()
    return render(request, 'AnalyseSentiment/index.html', {'form': form, 'sentiment': sentiment})
