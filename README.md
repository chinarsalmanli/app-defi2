# App-Pratique Defi 2

## Patient Survival Prediction

Les prédicteurs de la mortalité hospitalière des patients admis restent mal caractérisés.

L'objectif du défi 2 est de développer et valider `un modèle de prédiction de la mortalité hospitalière` toutes causes confondues chez les patients hospitalisés.

Les données sont disponibles dans le fichier `dataDefi2.csv.gz`

Ces données proviennent de l'initiative GOSSIS (Global Open Source Severity of Illness Score) du MIT et ont été collectées aux États-Unis en 2021.

Un cahier de variables décrivant brièvement les variables est également disponible (fihier `cahierVariables.csv`)

En résumé :

* Vous devez développer un modèle permettant de prédire le décès à l'hôpital de ces patients.
* Votre rendu est un code qui peut fonctionner sur un fichier de même structure que le fichier de données initial et qui retourne le statut vivant ou décédé (0/1) des sujets.

Un bonus sera attribué au groupe qui obtient la meilleure performance (exactitude, puis VPP (rappel) en cas d'égalité) sur le set de validation

Les groupes sont définis dans le fichier `groupes.csv`
