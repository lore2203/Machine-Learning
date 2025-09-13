# iris_linear_ovr.py
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Cargar dataset (numpy arrays para evitar warnings de feature names)
iris = load_iris()
X = iris.data     # shape (150,4)
y = iris.target   # 0=setosa,1=versicolor,2=virginica


# 1) Evaluación con train/test split (clasifica todas las muestras del test set)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# etiquetas binarias One-vs-Rest para el train
ys_train = (y_train == 0).astype(int)
yv_train = (y_train == 1).astype(int)
yvi_train = (y_train == 2).astype(int)

# entrenar 3 modelos de regresión lineal (uno por clase)
m_set = LinearRegression().fit(X_train, ys_train)
m_ver = LinearRegression().fit(X_train, yv_train)
m_vir = LinearRegression().fit(X_train, yvi_train)

# predecir (scores) en el conjunto de prueba
s_set = m_set.predict(X_test)
s_ver = m_ver.predict(X_test)
s_vir = m_vir.predict(X_test)

# ensamblar scores y elegir la clase con mayor valor
scores_test = np.vstack([s_set, s_ver, s_vir]).T
y_pred_test = np.argmax(scores_test, axis=1)

# métricas en test set
print("=== Resultados en el conjunto de prueba (unseen) ===")
print("Accuracy (test):", accuracy_score(y_test, y_pred_test))
print("\nClassification report (test):")
print(classification_report(y_test, y_pred_test, target_names=iris.target_names))
print("Confusion matrix (test):")
print(confusion_matrix(y_test, y_pred_test))

# mostrar predicciones por muestra (test set)
print("\nPredicciones por muestra (test set):")
for i, (x, true, pred, a,b,c) in enumerate(zip(X_test, y_test, y_pred_test, s_set, s_ver, s_vir)):
    print(f"{i:02d}: features={np.round(x,2)} -> True={iris.target_names[true]:9s} Pred={iris.target_names[pred]:9s} scores=[{a:.3f},{b:.3f},{c:.3f}]")

# 2) Clasificar las 150 muestras con validación cruzada 5-fold (predicciones out-of-fold)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores_cv = np.zeros((len(X), 3))  #3 scores por muestra (out-of-fold)

for train_idx, test_idx in kf.split(X, y):
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr = y[train_idx]
    # etiquetas binarias para el fold
    ys_tr = (y_tr == 0).astype(int)
    yv_tr = (y_tr == 1).astype(int)
    yvi_tr = (y_tr == 2).astype(int)
    # entrenar en el fold
    m0 = LinearRegression().fit(X_tr, ys_tr)
    m1 = LinearRegression().fit(X_tr, yv_tr)
    m2 = LinearRegression().fit(X_tr, yvi_tr)
    # predecir scores para las muestras del test_idx (out-of-fold)
    scores_cv[test_idx, 0] = m0.predict(X_te)
    scores_cv[test_idx, 1] = m1.predict(X_te)
    scores_cv[test_idx, 2] = m2.predict(X_te)

# decisión final por muestra usando los scores out-of-fold
y_pred_cv = np.argmax(scores_cv, axis=1)

print("\n\n=== Resultados usando 5-fold CV (predicciones out-of-fold para las 150 muestras) ===")
print("Accuracy (CV, todas las muestras):", accuracy_score(y, y_pred_cv))
print("\nClassification report (CV, todas las muestras):")
print(classification_report(y, y_pred_cv, target_names=iris.target_names))
print("Confusion matrix (CV, todas las muestras):")
print(confusion_matrix(y, y_pred_cv))

# mostrar predicciones out-of-fold para todas las 150 muestras
print("\nPredicciones out-of-fold (todas las muestras):")
for i in range(len(X)):
    sc0, sc1, sc2 = scores_cv[i]
    print(f"{i+1:03d}: features={np.round(X[i],2)} -> True={iris.target_names[y[i]]:9s} Pred={iris.target_names[y_pred_cv[i]]:9s} scores=[{sc0:.3f},{sc1:.3f},{sc2:.3f}]")

