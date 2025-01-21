import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
# 1. Charger le fichier CSV
# Remplacez 'movies_reviews.csv' par le nom de votre fichier CSV
file_path = 'movies_reviews.csv'
data = pd.read_csv(file_path)

# 2. Examiner les premières lignes
print("Premières lignes du dataset :")
print(data.head())

# 3. Vérifier la structure et les informations des données
print("\nStructure du dataset :")
print(data.info())

# 4. Statistiques de base pour les étiquettes
print("\nDistribution des étiquettes (1 = Positif, 0 = Négatif) :")
print(data['sentiment'].value_counts())

# 5. Vérifier les valeurs manquantes
print("\nValeurs manquantes :")
print(data.isnull().sum())

# Télécharger les stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))  # Changez pour 'english' si nécessaire

def nettoyer_texte(texte):
    # Supprimer les caractères spéciaux et chiffres
    texte = re.sub(r'[^a-zA-Z\s]', '', texte)
    # Convertir en minuscules
    texte = texte.lower()
    # Supprimer les mots vides
    mots = texte.split()
    mots = [mot for mot in mots if mot not in stop_words]
    return ' '.join(mots)

# Appliquer le nettoyage à la colonne des avis
data['cleaned_review'] = data['review'].apply(nettoyer_texte)

# Afficher quelques lignes nettoyées
print("Exemples de textes nettoyés :")
print(data[['review', 'cleaned_review']].head())

# Les données (caractéristiques) sont dans la matrice TF-IDF
# Les étiquettes sont dans la colonne 'label'
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['cleaned_review'])  # Assurez-vous d'avoir exécuté la transformation TF-IDF
y = data['sentiment']  # Remplacez 'label' par le nom exact de la colonne d'étiquettes si différent

# Diviser les données en ensemble d'entraînement (80%) et de test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Afficher la taille des ensembles
print("Taille de l'ensemble d'entraînement :", X_train.shape[0])
print("Taille de l'ensemble de test :", X_test.shape[0])


# Initialisation du modèle de régression logistique
model = LogisticRegression()

# Entraînement du modèle avec l'ensemble d'entraînement
model.fit(X_train, y_train)

print("Modèle de régression logistique entraîné avec succès.")


# Initialisation du modèle k-NN avec k=5 voisins
knn_model = KNeighborsClassifier(n_neighbors=5)

# Entraînement du modèle k-NN avec l'ensemble d'entraînement
knn_model.fit(X_train, y_train)

print("Modèle k-NN entraîné avec succès.")



# 1. Faire des prédictions sur l'ensemble de test
y_pred = model.predict(X_test)  # Remplacez "model" par knn_model si vous utilisez k-NN

# 2. Calculer les métriques
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Évaluation des performances :")
print(f"Précision (Accuracy) : {accuracy:.2f}")
print(f"Précision (Precision) : {precision:.2f}")
print(f"Rappel (Recall) : {recall:.2f}")
print(f"Score F1 : {f1:.2f}")

# 3. Matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred)

# Afficher la matrice de confusion
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Négatif', 'Positif'], yticklabels=['Négatif', 'Positif'])
plt.xlabel('Prédictions')
plt.ylabel('Vérités')
plt.title('Matrice de confusion')
plt.show()

# 4. Rapport détaillé
print("\nRapport détaillé :")
print(classification_report(y_test, y_pred, target_names=['Négatif', 'Positif']))

import pickle

# Sauvegarder le modèle
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Sauvegarder le vectoriseur
with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)
