
!pip install pandas numpy scikit-learn imbalanced-learn

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

from sklearn.utils import resample

data = pd.read_csv("/content/Creditcard_data.csv")

X = data.drop("Class", axis=1)
y = data["Class"]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

models = {
    "M1": LogisticRegression(max_iter=1000),
    "M2": DecisionTreeClassifier(),
    "M3": RandomForestClassifier(n_estimators=100),
    "M4": SVC(),
    "M5": KNeighborsClassifier()
}


sampling_methods = [
    "Bootstrap",
    "RandomUnder",
    "RandomOver",
    "KFold",
    "Stratified"
]

results = pd.DataFrame(index=models.keys(), columns=sampling_methods)

X_boot, y_boot = resample(
    X_scaled,
    y,
    replace=True,
    n_samples=len(y),
    random_state=42
)

rus = RandomUnderSampler(random_state=42)
X_under, y_under = rus.fit_resample(X_scaled, y)


ros = RandomOverSampler(random_state=42)
X_over, y_over = ros.fit_resample(X_scaled, y)


sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)

for train_idx, test_idx in sss.split(X_scaled, y):
    X_strat_train, X_strat_test = X_scaled[train_idx], X_scaled[test_idx]
    y_strat_train, y_strat_test = y.iloc[train_idx], y.iloc[test_idx]

def train_test_eval(Xd, yd, model):
    X_train, X_test, y_train, y_test = train_test_split(
        Xd,
        yd,
        test_size=0.3,
        random_state=42
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds) * 100

for m_name, model in models.items():
    results.loc[m_name, "Bootstrap"] = round(train_test_eval(X_boot, y_boot, model), 2)
    results.loc[m_name, "RandomUnder"] = round(train_test_eval(X_under, y_under, model), 2)
    results.loc[m_name, "RandomOver"] = round(train_test_eval(X_over, y_over, model), 2)


    model.fit(X_strat_train, y_strat_train)
    preds = model.predict(X_strat_test)
    results.loc[m_name, "Stratified"] = round(
        accuracy_score(y_strat_test, preds) * 100, 2
    )

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for m_name, model in models.items():
    scores = []
    for train_idx, test_idx in kf.split(X_scaled):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        scores.append(accuracy_score(y_test, preds))

    results.loc[m_name, "KFold"] = round(np.mean(scores) * 100, 2)

print("\n Accuracy comparison table is as follows :\n")
print(results)

print("\n Best sampling method per model is :\n")
for model in results.index:
    best_method = results.loc[model].astype(float).idxmax()
    best_score = results.loc[model].astype(float).max()
    print(f"{model} → {best_method} ({best_score}%)")
