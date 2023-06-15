from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def train_models(X_train, y_train, random_state=42):
    catboost_clf = CatBoostClassifier(silent=True, random_state=random_state)
    catboost_clf.fit(X_train, y_train)

    lightgbm_clf = LGBMClassifier(random_state=random_state)
    lightgbm_clf.fit(X_train, y_train)

    random_forest_clf = RandomForestClassifier(random_state=random_state)
    random_forest_clf.fit(X_train, y_train)

    return catboost_clf, lightgbm_clf, random_forest_clf