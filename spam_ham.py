# ================================
# 1. Importar librerías
# ================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, accuracy_score
)

# ================================
# 2. Cargar dataset
# ================================
# Asegúrate de que tu CSV tenga la columna "spam" (1=spam, 0=no spam)
df = pd.read_csv("C:/Users/Lore3/OneDrive/Escritorio/Octavo/Dataset emails machine.csv")

# Features (las 10 columnas de entrada) y Target (spam/no spam)
X = df.drop(columns=["spam_label", "email_id","sender","subject","body_excerpt", "sender_domain_reputation"])
y = df["spam_label"].map({"ham": 0, "spam":1})

# ================================
# 3. Separar datos en train y test
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ================================
# 4. Entrenar modelo (Regresión Logística)
# ================================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ================================
# 5. Probabilidades y umbral
# ================================
# Probabilidades de que sea SPAM
y_probs = model.predict_proba(X_test)[:, 1]

# Predicción con el umbral por defecto (0.5)
threshold = 0.5
y_pred = (y_probs >= threshold).astype(int)

# ================================
# 6. Métricas
# ================================
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-Score:", f1_score(y_test, y_pred))
print("\nReporte completo:\n", classification_report(y_test, y_pred))

# ================================
# 7. Matriz de confusión
# ================================
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Spam", "Spam"],
            yticklabels=["No Spam", "Spam"])
plt.title("Matriz de Confusión")
plt.xlabel("Predicción")
plt.ylabel("Realidad")
plt.show()

# ================================
# 8. Importancia de features
# ================================
coef = model.coef_[0]
importance = np.abs(coef) / np.sum(np.abs(coef)) * 100  # en %
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importancia (%)": importance
}).sort_values(by="Importancia (%)", ascending=False)

print("\nImportancia de los features (%):\n", feature_importance)

# Gráfico
plt.figure(figsize=(8,5))
sns.barplot(data=feature_importance, x="Importancia (%)", y="Feature", palette="viridis")
plt.title("Importancia de los Features en el Modelo (%)")
plt.show()

# ================================
# 9. Correlación
# ================================
plt.figure(figsize=(10,8))
corr = df.select_dtypes(include=["int64", "float64"]).corr()

# Graficar el mapa de calor
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Mapa de Correlación entre Features Numéricos")
plt.show()
