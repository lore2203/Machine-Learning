import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import zscore

# Cargar dataset
df = pd.read_csv(r"C:\Users\Lore3\OneDrive\Escritorio\Octavo\Machine\Dataset emails machine.csv")


# Corregir nombre de la columna y limpiar etiquetas
df = df.rename(columns={df.columns[-1]: "spam_label"})
df["spam_label"] = df["spam_label"].str.replace(";", "", regex=False)
df["spam_label"] = df["spam_label"].map({"spam": 1, "ham": 0})

# Eliminar filas con etiquetas faltantes
df = df.dropna(subset=["spam_label"])

# Seleccionar características numéricas
X = df[["subject_word_count", "body_word_count", "num_links", 
        "num_attachments", "contains_spam_words_count",
        "subject_length_chars", "body_uppercase_ratio", 
        "avg_word_length_body", "special_char_count_subject"]]
y = df["spam_label"].astype(int)

# Guardar resultados
results = {"accuracy": [], "f1": []}

# Ejecutar 50 veces
for i in range(50):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=i
    )
    
    model = DecisionTreeClassifier(random_state=i)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results["accuracy"].append(acc)
    results["f1"].append(f1)

# Convertir resultados a DataFrame
results_df = pd.DataFrame(results)

# Calcular z-score
results_df["accuracy_z"] = zscore(results_df["accuracy"])
results_df["f1_z"] = zscore(results_df["f1"])

print(results_df.describe())

# Graficar distribuciones
plt.figure(figsize=(10,5))
plt.plot(results_df["accuracy"], label="Accuracy")
plt.plot(results_df["f1"], label="F1-score")
plt.xlabel("Ejecución")
plt.ylabel("Valor")
plt.title("Resultados en 50 ejecuciones")
plt.legend()
plt.show()

# Histograma de métricas
results_df[["accuracy", "f1"]].hist(bins=10, figsize=(10,4))
plt.suptitle("Distribución de Accuracy y F1-score")
plt.show()

# Histograma de z-scores
results_df[["accuracy_z", "f1_z"]].hist(bins=10, figsize=(10,4))
plt.suptitle("Distribución de Z-scores")
plt.show()
