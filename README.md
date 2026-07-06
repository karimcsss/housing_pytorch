# Prédicteur de Prix Immobilier Interactif (MLP & PyTorch)

Ce projet implémente une chaîne complète de Machine Learning, allant du nettoyage et de la normalisation de données réelles jusqu'à la création d'une interface utilisateur interactive. Il s'appuie sur le jeu de données California Housing.

## Fonctionnalités du Projet

- Traitement de Données (Pandas) : Nettoyage des valeurs manquantes et normalisation Min-Max optimisée grâce au mécanisme de Broadcasting.
- Deep Learning (PyTorch) : Architecture Perceptron Multicouche (MLP) utilisant des couches linéaires (nn.Linear) et des activations ReLU, entraînée avec une fonction de perte de régression (MSELoss).
- Visualisation (Matplotlib) : Génération et sauvegarde automatique d'un graphique (performances_mlp.png) présentant la courbe de perte au fil des époques ainsi qu'un nuage de points (prix réels vs prix prédits sur un échantillon sécurisé).
- Interface Utilisateur (Gradio) : Une application web locale dotée de sliders interactifs permettant de tester l'inférence du modèle PyTorch en temps réel.


Déroulement de l'exécution :
1. Le script charge les données et applique la normalisation vectorielle.
2. Le réseau PyTorch s'entraîne sur 100 époques.
3. Une fenêtre Matplotlib s'ouvre pour afficher l'évaluation (le graphique est également exporté localement sous le nom performances_mlp.png).
4. Fermez la fenêtre du graphique pour déclencher automatiquement le lancement de l'interface Gradio dans votre navigateur internet.



Le modèle utilise une structure simple et robuste adaptée à la régression :
- Entrée : 3 caractéristiques (median_income, housing_median_age, total_rooms)
- Couche Cachée 1 : 16 neurones + Activation ReLU
- Couche Cachée 2 : 8 neurones + Activation ReLU
- Sortie : 1 neurone (Prix Médian Prédit)
