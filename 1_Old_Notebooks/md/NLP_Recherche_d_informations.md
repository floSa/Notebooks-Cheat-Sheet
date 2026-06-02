---
jupyter:
  jupytext:
    notebook_metadata_filter: -jupytext.text_representation.jupytext_version
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
  kernelspec:
    display_name: Python 3
    name: python3
---

```python colab={"base_uri": "https://localhost:8080/"} id="XFgnrIm7OdCM" executionInfo={"status": "ok", "timestamp": 1690207689203, "user_tz": -120, "elapsed": 9591, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="c125acee-4901-4a2d-b384-bb1c9cf95c3d"
pip install gensim
```

```python colab={"base_uri": "https://localhost:8080/"} id="b0EwbHOOOlGI" executionInfo={"status": "ok", "timestamp": 1690207795947, "user_tz": -120, "elapsed": 280, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="faa9db75-4d84-4c12-f554-6fd86dd02460"
from gensim import corpora, models, similarities

# Exemple de dataset avec ID et contenu d'article
dataset = [
    {'ID': 1, 'content': "Ceci est un article sur les chats."},
    {'ID': 2, 'content': "Les chiens et les chats sont des animaux domestiques populaires."},
    {'ID': 3, 'content': "Comment prendre soin d'un chat domestique."},
    {'ID': 4, 'content': "Les oiseaux sont de magnifiques créatures volantes."},
]

# Prétraitement des articles (tokenization, suppression des mots vides, etc.)
def preprocess(text):
    # À implémenter : Appliquer ici les étapes de prétraitement nécessaires (ex : suppression de mots vides, tokenization, etc.)
    return text.split()

# Création du corpus et du dictionnaire
corpus = [preprocess(article['content']) for article in dataset]
dictionary = corpora.Dictionary(corpus)

# Création du modèle TF-IDF
tfidf = models.TfidfModel(dictionary=dictionary)
corpus_tfidf = [tfidf[dictionary.doc2bow(doc)] for doc in corpus]

# Création de l'index pour la recherche
index = similarities.MatrixSimilarity(corpus_tfidf)

# Fonction pour trouver les articles pertinents à partir d'une phrase de recherche
def search(query, top_n=3):
    query_vec = tfidf[dictionary.doc2bow(preprocess(query))]
    similarities = index[query_vec]
    similar_indices = similarities.argsort()[::-1][:top_n]
    return [dataset[i]['ID'] for i in similar_indices]

# Exemple d'utilisation
search_query = "que dois-je faire pour garder mon félin à la maison?"
relevant_articles = search(search_query)
print("Articles pertinents :", relevant_articles)

```

```python colab={"base_uri": "https://localhost:8080/"} id="l9ubJnXWQMof" executionInfo={"status": "ok", "timestamp": 1690208227377, "user_tz": -120, "elapsed": 6947, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="e1f95d93-d719-4eda-d3ea-33d1cf9e4598"
import spacy.cli
spacy.cli.download("fr_core_news_sm")
```

```python colab={"base_uri": "https://localhost:8080/"} id="Sj-xXKQGOpY3" executionInfo={"status": "ok", "timestamp": 1690208294324, "user_tz": -120, "elapsed": 2393, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="3ae9354f-f5df-4bad-e345-4eb54e53856a"
import spacy
from sklearn.metrics.pairwise import cosine_similarity

# Exemple de dataset avec ID et contenu d'article
dataset = [
    {'ID': 1, 'content': "Ceci est un article sur les chats."},
    {'ID': 2, 'content': "Les chiens et les chats sont des animaux domestiques populaires."},
    {'ID': 3, 'content': "Comment prendre soin d'un chat domestique."},
    {'ID': 4, 'content': "Les oiseaux sont de magnifiques créatures volantes."},
]

# Chargement du modèle SpaCy
nlp = spacy.load("fr_core_news_sm")

# Fonction pour trouver les articles pertinents à partir d'une phrase de recherche
def search(query, top_n=3):
    query_doc = nlp(query)
    article_docs = [nlp(article['content']) for article in dataset]
    similarities = cosine_similarity(query_doc.vector.reshape(1, -1), [doc.vector for doc in article_docs])
    similar_indices = similarities.argsort()[0][::-1][:top_n]
    return [dataset[i]['ID'] for i in similar_indices]

# Exemple d'utilisation
search_query = "que dois-je faire pour garder mon félin à la maison?"
relevant_articles = search(search_query)
print("Articles pertinents :", relevant_articles)

```

```python colab={"base_uri": "https://localhost:8080/"} id="CATlQQ_AWMVS" executionInfo={"status": "ok", "timestamp": 1690209684746, "user_tz": -120, "elapsed": 16486, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="cc810dfc-c678-4eb1-d725-72cec1e84374"
pip install transformers
```

```python id="-Ry1jx1TPfu3"
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Exemple de dataset avec ID et contenu d'article
dataset = [
    {'ID': 1, 'content': "Ceci est un article sur les chats."},
    {'ID': 2, 'content': "Les chiens et les chats sont des animaux domestiques populaires."},
    {'ID': 3, 'content': "Comment prendre soin d'un chat domestique."},
    {'ID': 4, 'content': "Les oiseaux sont de magnifiques créatures volantes."},
]

# Chargement du modèle BERT et du tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Fonction pour encoder une phrase avec BERT avec padding
def encode_sentence(sentence, max_length=64):
    tokens = tokenizer.encode(sentence, add_special_tokens=True, return_tensors="pt", max_length=max_length, truncation=True, padding='max_length')
    with torch.no_grad():
        model_output = model(tokens)[0]
    return model_output

# Fonction pour trouver les articles pertinents à partir d'une phrase de recherche
def search(query, top_n=3):
    max_length = max(len(tokenizer.encode(article['content'])) for article in dataset)
    query_encoding = encode_sentence(query, max_length=max_length)
    article_encodings = [encode_sentence(article['content'], max_length=max_length) for article in dataset]
    similarities = cosine_similarity(query_encoding.view(1, -1), torch.stack(article_encodings).view(len(article_encodings), -1))
    similar_indices = similarities.argsort()[0][::-1][:top_n]
    return [dataset[i]['ID'] for i in similar_indices]

```

```python colab={"base_uri": "https://localhost:8080/"} id="71H681vpWC4R" executionInfo={"status": "ok", "timestamp": 1690210276485, "user_tz": -120, "elapsed": 2708, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="5125701f-a3c4-4a23-9718-472b08b881a3"
# Exemple d'utilisation
search_query = "que dois-je faire pour garder mon félin à la maison?"
relevant_articles = search(search_query)
print("Articles pertinents :", relevant_articles)
```

```python id="dizOcWZKW8yX"
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Exemple de dataset avec ID et contenu d'article
dataset = [
    {'ID': 1, 'content': "Airbus est une belle entreprise"},
    {'ID': 2, 'content': "j'ai acheter un A380, voici mon retour sur ce magnifique appareil"},
    {'ID': 3, 'content': "j'ai acheter flight simulator je vous donne mon avis"},
    {'ID': 4, 'content': "Les oiseaux sont de magnifiques créatures volantes"},
    {'ID': 5, 'content': "je vous explque comment faire des avions en papier"},
]

# Chargement du modèle BERT et du tokenizer spécifiques
model_name = "bert-large-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Fonction pour encoder une phrase avec BERT avec padding
def encode_sentence(sentence, max_length=32):
    tokens = tokenizer.encode(sentence, add_special_tokens=True, return_tensors="pt", max_length=max_length, truncation=True, padding='max_length')
    with torch.no_grad():
        model_output = model(tokens)[0]
    return model_output

# Fonction pour trouver les articles pertinents à partir d'une phrase de recherche
def search(query, top_n=3):
    max_length = max(len(tokenizer.encode(article['content'])) for article in dataset)
    query_encoding = encode_sentence(query, max_length=max_length)
    article_encodings = [encode_sentence(article['content'], max_length=max_length) for article in dataset]
    similarities = cosine_similarity(query_encoding.view(1, -1), torch.stack(article_encodings).view(len(article_encodings), -1))
    similar_indices = similarities.argsort()[0][::-1][:top_n]
    return [dataset[i]['ID'] for i in similar_indices]

```

```python colab={"base_uri": "https://localhost:8080/"} id="OrIaD9H3ZPDb" executionInfo={"status": "ok", "timestamp": 1690211680527, "user_tz": -120, "elapsed": 3512, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="f51a6938-87c8-4cd3-ebf3-bd2f3b826d07"
# Exemple d'utilisation
search_query = "retourne moi les article sur l'aviation"
relevant_articles = search(search_query,5)
print("Articles pertinents :", relevant_articles)
```

```python id="FvTINunBZpgx" colab={"base_uri": "https://localhost:8080/", "height": 223} executionInfo={"status": "ok", "timestamp": 1703003953514, "user_tz": -60, "elapsed": 3683, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="837e1807-5dd3-4180-8787-3771caa5c726"
# Importation de la bibliothèque pandas
import pandas as pd

# Liste des profils générés
profils = [
    "Je suis un DevOps avec 8 ans d'expérience. Mon expertise principale est dans l'écosystème Docker et Kubernetes. J'ai travaillé chez Microsoft pendant 3 ans, où j'ai consolidé mes compétences en automatisation des déploiements.",
    "En tant que Data Engineer depuis 5 ans, je me spécialise dans le traitement des données massives avec Apache Spark. Mon langage de prédilection est Python, et j'ai travaillé chez Amazon en construisant des pipelines de données robustes.",
    "En tant que Data Scientist, j'ai obtenu mon diplôme d'ingénieur en 2020. Je suis expert en Python, avec 4 ans d'expérience en utilisant des bibliothèques comme TensorFlow et scikit-learn. J'ai travaillé chez Google, où j'ai développé des modèles de machine learning pour améliorer la recommandation de produits.",
    "En tant que Back-End Developer, je maitrise Java depuis 6 ans. J'ai travaillé chez Facebook, contribuant au développement de services hautement évolutifs. Mon expertise inclut Spring Boot et la conception d'API RESTful.",
    "Front-End Developer passionné depuis 7 ans, je me spécialise dans React et Vue.js. J'ai une expérience significative chez Airbnb, où j'ai contribué à l'interface utilisateur de leur plateforme de réservation.",
    "Je suis un DevOps expérimenté avec 10 ans d'expérience. J'ai une expertise approfondie dans l'automatisation des déploiements avec Jenkins et Ansible. J'ai travaillé chez Amazon Web Services, optimisant les processus de déploiement.",
    "En tant que Data Engineer depuis 6 ans, je me spécialise dans la construction d'architectures de données évolutives avec Apache Kafka. J'ai travaillé chez Microsoft, où j'ai conçu des pipelines de streaming en temps réel.",
    "En tant que Data Scientist, je possède un doctorat en sciences des données. Je suis expert en R et Python, avec une expérience de 5 ans en traitement du langage naturel. J'ai travaillé chez IBM, développant des modèles de prédiction pour le secteur financier.",
    "En tant que Back-End Developer depuis 8 ans, je maitrise C# et ASP.NET Core. J'ai travaillé chez Google, contribuant au développement de services backend hautement performants.",
    "Front-End Developer créatif avec 4 ans d'expérience, je suis passionné par l'utilisation de technologies comme React et Angular. J'ai travaillé chez Spotify, améliorant l'expérience utilisateur de leur application de streaming.",
    "DevOps chevronné avec 12 ans d'expérience. Je me spécialise dans l'orchestration des conteneurs avec Kubernetes et Helm. J'ai contribué à l'automatisation des déploiements chez Netflix pendant 6 ans.",
    "Data Engineer passionné par les données distribuées. Fort de 7 ans d'expérience, je maîtrise Apache Flink pour le traitement de flux en temps réel. Mon parcours inclut un passage chez Twitter, où j'ai optimisé les pipelines de données.",
    "Data Scientist créatif et innovant. Titulaire d'un master en statistiques, j'ai 3 ans d'expérience en utilisant R et Python pour développer des modèles de recommandation. J'ai travaillé chez Airbnb, améliorant la personnalisation des suggestions.",
    "Back-End Developer expérimenté avec 9 ans d'expertise en Java et Spring Boot. J'ai travaillé chez LinkedIn, contribuant au développement de services backend haute performance et à la gestion des bases de données.",
    "Front-End Developer enthousiaste avec 5 ans d'expérience. Maîtrisant React et Vue.js, j'ai travaillé chez Microsoft, améliorant l'interface utilisateur de leurs applications web grand public.",
    "DevOps stratégique avec 11 ans d'expérience. J'ai dirigé des initiatives d'automatisation chez Google Cloud, en mettant en œuvre des solutions basées sur Terraform. Ma passion pour l'efficacité opérationnelle a considérablement réduit les temps de déploiement.",
    "Data Engineer axé sur la scalabilité des données. J'ai 8 ans d'expérience dans la conception de systèmes distribués avec Apache Hadoop. Mon expertise chez Uber a contribué à la gestion efficace des données à grande échelle.",
    "Data Scientist créatif avec 4 ans d'expérience. Mon domaine d'expertise inclut le traitement du langage naturel avec spaCy et la classification d'image avec TensorFlow. J'ai travaillé chez Tesla, appliquant l'IA pour améliorer la conduite autonome.",
    "Back-End Developer spécialisé dans les microservices. J'ai 6 ans d'expérience avec Go et j'ai contribué au développement d'API RESTful chez Amazon Web Services. Ma passion pour l'évolutivité a optimisé les performances des services.",
    "Front-End Developer passionné par l'esthétique et la performance. Avec 7 ans d'expérience, j'ai travaillé chez Apple, contribuant à l'expérience utilisateur des applications iOS avec SwiftUI et UIKit.",
    "Architecte Cloud avec 15 ans d'expérience. J'ai conçu des solutions cloud évolutives chez AWS, utilisant des services tels que EC2, S3 et Lambda. Mon expertise a contribué à l'optimisation des architectures cloud chez Microsoft Azure.",
    "Responsable de la sécurité informatique avec 10 ans d'expérience. J'ai dirigé des initiatives de cybersécurité chez IBM, mettant en place des politiques de sécurité robustes et des tests d'intrusion avancés.",
    "Administrateur système chevronné avec 12 ans d'expérience. J'ai géré des infrastructures complexes chez Google Cloud, optimisant les performances et garantissant une disponibilité maximale.",
    "Ingénieur réseau passionné par l'optimisation des performances. Fort de 8 ans d'expérience chez Cisco, j'ai conçu des architectures réseau haute disponibilité pour des entreprises de premier plan.",
    "Analyste de données stratégique avec 6 ans d'expérience. J'ai travaillé chez Facebook, analysant les métriques clés pour orienter les décisions commerciales. Ma maîtrise de SQL et de Tableau a facilité des analyses approfondies.",
    "Ingénieur DevOps spécialisé dans l'intégration continue et le déploiement continu (CI/CD). Avec 9 ans d'expérience, j'ai automatisé les pipelines chez GitLab, améliorant la rapidité et la fiabilité des livraisons.",
    "Architecte logiciel orienté microservices. J'ai 11 ans d'expérience dans la conception d'architectures évolutives chez Netflix, utilisant des technologies telles que Spring Cloud et Docker.",
    "Consultant en intelligence artificielle. Fort de 7 ans d'expérience, j'ai conseillé des entreprises chez McKinsey sur l'adoption de l'IA, en développant des stratégies personnalisées et des solutions innovantes.",
    "Spécialiste en expérience utilisateur (UX) avec 8 ans d'expérience. J'ai travaillé chez Adobe, concevant des interfaces intuitives pour les applications de création graphique.",
    "Analyste en cybersécurité passionné par la prévention des menaces. Avec 5 ans d'expérience chez Palo Alto Networks, j'ai mis en place des stratégies de détection avancée pour protéger les réseaux d'entreprise.",
    "Architecte Visionnaire du Cloud avec 15 années d'expérience. J'ai sculpté des univers infinis chez AWS, fusionnant des étoiles EC2, des galaxies S3 et des comètes Lambda. Ma passion ? Construire des architectures interstellaires chez Microsoft Azure.",
    "Gardien du Sanctuaire de la Cybersécurité, je réside depuis 10 ans à la croisée des mondes numériques. Initié chez IBM, je forge des boucliers de sécurité impénétrables et exécute des danses complexes de tests d'intrusion.",
    "Administrateur Système, maestro des réseaux virtuels depuis 12 ans. Chez Google Cloud, je suis le chef d'orchestre d'une symphonie de bits, harmonisant les performances et veillant à ce que l'infrastructure soit toujours en plein concert.",
    "Ingénieur Réseau, explorateur des connexions cosmiques depuis 8 ans. Sur la planète Cisco, j'ai tracé des voies hyperespace pour des entreprises étoilées, assurant une communication intergalactique sans faille.",
    "Analyste de Données, alchimiste des informations chez Facebook depuis 6 ans. Je transforme des données brutes en formules magiques, utilisant SQL comme ma baguette et Tableau comme mon grimoire.",
    "Ingénieur DevOps, artisan de l'automatisation magique depuis 9 ans. Chez GitLab, mes formules CI/CD transforment les codes en potions déployables, apportant vitesse et fiabilité comme par magie.",
    "Architecte Logiciel, bâtisseur de royaumes microservices depuis 11 ans chez Netflix. Je façonne des architectures résistantes comme des forteresses, utilisant des sorts tels que Spring Cloud et Docker.",
    "Consultant en IA, astronaute des stratégies intelligentes depuis 7 ans. J'ai exploré les galaxies des affaires chez McKinsey, propulsant les entreprises vers des horizons nouveaux avec des solutions AI innovantes.",
    "Maître de l'Expérience Utilisateur (UX), artiste du design depuis 8 ans chez Adobe. Je sculpte des interfaces visuelles captivantes, élevant les logiciels au rang d'œuvres d'art numériques.",
    "Analyste en Cybersécurité, gardien du cyberespace depuis 5 ans chez Palo Alto Networks. J'utilise des stratégies avancées pour détecter et éliminer les menaces, tel un chevalier protégeant le royaume numérique.",
    "Data Scientist expérimenté, avec 8 ans dans le secteur médical. J'ai apporté ma contribution à l'oncologie de l'hôpital Saint-Louis, développant des modèles de prédiction pour la personnalisation des traitements.",
    "Développeur Full Stack avec 6 ans d'expérience dans la finance. J'ai œuvré au sein de l'équipe technologique de la Banque Royale, concevant des applications sécurisées pour la gestion des comptes clients.",
    "Ingénieur DevOps spécialisé dans l'industrie automobile. Pendant 7 ans chez Ford, j'ai automatisé les processus de déploiement, améliorant l'efficacité des chaînes de production logicielle.",
    "Responsable de la sécurité informatique dans le secteur gouvernemental. Avec 9 ans d'expérience, j'ai dirigé des initiatives de cybersécurité pour le département de la Défense, assurant la protection des données sensibles.",
    "Analyste de Données dans le domaine de l'énergie. Fort de 5 ans chez Chevron, j'ai analysé les données opérationnelles pour optimiser les performances des plates-formes pétrolières et gazières.",
    "Architecte Cloud avec une expertise dans le secteur de l'e-commerce. Pendant 10 ans chez Amazon, j'ai conçu des infrastructures cloud pour supporter le trafic massif des sites de vente en ligne.",
    "Développeur Front-End spécialisé dans le secteur du divertissement. J'ai contribué à l'expérience utilisateur de jeux vidéo chez Ubisoft, en développant des interfaces graphiques interactives.",
    "Ingénieur Réseau dans le domaine de l'aérospatiale. Avec 8 ans chez SpaceX, j'ai conçu des réseaux sécurisés pour assurer la communication entre les systèmes embarqués des fusées et des satellites.",
    "Consultant en Intelligence Artificielle pour le secteur de la santé. Avec 7 ans d'expérience chez Siemens Healthineers, j'ai développé des solutions d'IA pour améliorer les diagnostics médicaux.",
    "Développeur Back-End dans l'industrie agroalimentaire. J'ai contribué à la gestion des stocks chez Nestlé, en développant des systèmes backend pour optimiser la chaîne d'approvisionnement."
]

# Création d'un DataFrame avec une colonne "profils"
df = pd.DataFrame({"profils": profils})
print(df.shape)
df.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="AEhMJsxb8smG" executionInfo={"status": "ok", "timestamp": 1703004020475, "user_tz": -60, "elapsed": 20230, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="8b323690-6440-486e-a22b-5c6f0a1d9aad"
pip install transformers
```

```python colab={"base_uri": "https://localhost:8080/", "height": 509} id="DBuOhu-p8gjP" executionInfo={"status": "error", "timestamp": 1703005150533, "user_tz": -60, "elapsed": 5242, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="68bdf40d-786f-4657-d839-a8d46211d08f"
from transformers import pipeline

# Charger un modèle pré-entraîné BERT pour NLP
nlp = pipeline("text2text-generation", model="bert-base-uncased", tokenizer="bert-base-uncased")

# Exemple d'utilisation avec BERT
phrase_entree = "Je cherche développeur avec 5 ans d'expériences avec Python ou langage similaire"
resultats = nlp(phrase_entree, df['profils'].tolist())

# Récupérer les textes et les scores générés
textes_scores = [(resultat['generated_text'], resultat['score']) for resultat in resultats]

# Classer les textes par ordre de pertinence (du plus pertinent au moins pertinent)
textes_pertinents_tries = sorted(textes_scores, key=lambda x: x[1], reverse=True)

# Affichage des résultats classés
for texte, score in textes_pertinents_tries:
    print(f"Pertinence: {score:.4f} - Texte: {texte}")
```

```python id="sE8yPFbd9EBU"

```
