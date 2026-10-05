"""
Definició del model de predicció (RandomForest).

Tenir-lo en un fitxer a part permet canviar-lo en un sol lloc, compartit
per entrenament.py, main2.py i els optimitzadors.
"""

from sklearn.ensemble import RandomForestClassifier


def make_model():
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=2,
        max_features="log2",
        random_state=42,
    )
