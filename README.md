# Conception d'un pipeline MLOps pour l'estimation automatisé des ressources marines | Collaboration INRH & ENSIAS

## Sommaire

- [Video demo](#video-demo)
- [Description du projet](#description-du-projet)
- [Vue d'ensemble du Projet](#vue-densemble-du-projet)
- [Approche Scientifique & Modélisation](#approche-scientifique--modélisation)
  - [Résultats & Performances](#résultats--performances)
- [Architecture MLOps & Déploiement](#architecture-mlops--déploiement)
  - [Interface Utilisateur (Streamlit)](#interface-utilisateur-streamlit)
  - [Orchestration et Automatisation (Apache Airflow & Docker)](#orchestration-et-automatisation-apache-airflow--docker)
- [Technologies Utilisées](#technologies-utilisées)
- [Structure du Dépôt (Principal)](#structure-du-dépôt-principal).

---
## **Video demo**

<video src="https://github.com/floriansilvestre67/Conception-d-un-pipeline-MLOps-pour-l-estimation-automatis-e-des-ressources-marines./raw/refs/heads/main/data/content_readme/demo_courte_PFA.mp4" controls width="600">
</video>

## **Description du projet** 
Projet de fin d'année (PFA) réalisé en collaboration avec **l'Institut National de Recherche Halieutique (INRH) du Maroc**.
L'objectif est de prédire la biomasse halieutique de l'anchois à partir de données environnementales et acoustiques complexes collectées lors de campagnes scientifiques en mer entre 2015 et 2021.

Il s'agit d'un projet **end-to-end** complet, couvrant tout le cycle de la Data Science : depuis l'exploration et le nettoyage des données jusqu'au déploiement d'une application de prédiction en temps réel et la mise en place d'un pipeline MLOps conteneurisé.

---

## Vue d'ensemble du Projet

La variable cible à prédire est le **coefficient de rétrodiffusion acoustique NASC** (*Nautical Area Scattering Coefficient*), qui sert de proxy pour estimer la densité de poissons. Ce jeu de données présentait des défis scientifiques majeurs:
* **Zero-inflation massive :** 81 % des observations ont une valeur nulle (absence d'anchois).
* **Asymétrie extrême :** Variabilité colossale sur les zones de présence (NASC variant de 0,5 à plus de 51 000).
* **Multicolinéarité :** Forte redondance entre les variables chimiques et biologiques marines.

---

## Approche Scientifique & Modélisation

Pour surmonter la problématique des 81 % de zéros (absence de poissons) , l'architecture repose sur un **Modèle Hurdle en deux phases**:

1. **Phase 1 (Classification binaire) :** Détection de la présence ou de l'absence d'anchois.
2. **Phase 2 (Régression conditionnelle) :** Estimation quantitative de la biomasse uniquement sur les zones de présence détectées, en utilisant la cible transformée via *Yeo-Johnson* pour corriger l'asymétrie.

### Résultats & Performances
Après une phase intensive de comparaison de modèles (modèles linéaires, géométriques, deep learning et ensembles), l'algorithme **Hist Gradient Boosting** s'est imposé sur les deux phases. Les hyperparamètres des 2 modèles ont été optimisé via **Optuna (TPE)** :

* **Classification (Présence) :** **F1-Score de 0,596** | ROC-AUC de 0,912 | Accuracy (présence) de 0.73 | Recall (presence) 0.50.
* **Régression (Quantité) :** **$R^2$ de 0,490



** | RMSE de 0,356 | MAE de 0,235.

>  **Variables clés identifiées :** L'analyse de l'importance par permutation montre que la présence des anchois est fortement gouvernée par le **fer dissous (Fe)** (proxy de la production primaire), tandis que la quantité de biomasse est un phénomène purement **saisonnier** (capturé par l'encodage cyclique des mois).

---

##  Architecture MLOps & Déploiement

Le projet bascule d'un simple code expérimental en environnement Notebook vers un système robuste prêt pour la production:

### Interface Utilisateur (Streamlit)
Une interface web interactive développée avec **Streamlit** et **Plotly** permet aux utilisateurs non spécialistes d'exploiter les modèles:
* Saisie des conditions environnementales selon 3 modes (Simple, Avancé et Personalisé).
* Visualisation dynamique en temps réel de la probabilité de présence et de la biomasse estimée.
* Cartographie interactive de la distribution spatiale des anchois et graphiques de l'importance des variables.

### Orchestration et Automatisation (Apache Airflow & Docker)
L'intégralité du cycle de réentraînement est automatisée sous forme de **DAG Airflow** séparé en 5 taches réparti sur 4 conteneurs **Docker** interconnectés (`PostgreSQL`, `Airflow-Scheduler`, `Airflow-Webserver`, `Streamlit`):
1. `preprocess` : Nettoyage automatique et feature engineering (encodage cyclique, suppression des doublons/outliers).
2. `optimize` : Fine-tuning automatisé des modèles via Optuna.
3 & 4. `train_classifier` & `train_regressor` : Entraînement séquentiel du pipeline Hurdle.
5. `evaluate` : Calcul des métriques et export automatique vers Streamlit.

*Grâce à cette infrastructure, l'intégration d'une nouvelle campagne acoustique en mer se fait sans aucune intervention technique humaine, par simple dépôt du nouveau fichier `.xls`.*

---

## Technologies Utilisées

* **Langage & Manipulation Data :** Python, Pandas, NumPy.
* **Machine Learning :** Scikit-learn (Hist Gradient Boosting), Optuna.
* **Visualisation :** Streamlit, Plotly, Matplotlib.
* **MLOps & DevOps :** Apache Airflow 2.9, Docker, Docker.

---

## Structure du Dépôt (Principal)

```text
anchois-mlops/
├── airflow/                    # Image Docker Airflow + dépendances
├── dags/                       # DAG Airflow (pipeline 5 étapes)
├── data/raw/                   # Données brutes (réelles ignorées, fake incluse)
├── pipeline/                   # Scripts ML (preprocess → optimize → train → evaluate)
├── streamlit_app/              # Interface de visualisation et prédiction
├── docker-compose.yml          # Orchestration des 5 conteneurs
└── README.md

Rq : Afin de conserver la confidentialité des données sources sur Github c'est une fichier de fausses données qui a été 
