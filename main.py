import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Voglio usare la Logistic Regression, quindi in precedenza ho editato il file csv per avere poisonous ed edible
# con valori di 1 e 0; per le altre features ho dato valori progressivi (da 0 a 100) in base alla varietà dei valori
# che potevano assumere; ad esempio se "lunghezza gambo" assumeva 6 valori allora l'intervallo di valori possibili è
# 0, 20, 40, 60, 80 e 100.
# ->> Facendo ciò mi dava un error "Impossibile fare divisione per zero", questo perchè la colonna veil-type
# Può assumere un solo valore, quindi l'ho eliminata come prima cosa.

# Leggo i dati
dati = pd.read_csv("/home/jomin/Desktop/mushrooms_values_edited.csv")

### ANALISI DATI ###

# Vediamo se manca qualche dato
train_na = (dati.isnull().sum() / len(dati)) * 100
train_na = train_na.drop(train_na[train_na == 0].index).sort_values(ascending=False)[:30]
miss_train = pd.DataFrame({'Valori Mancanti': train_na})
print(miss_train.head())

features = ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment',
            'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
            'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-color',
            'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']

for feature in features:
    # Come è distribuita la velenosità a secondo della feature?
    print(dati[[feature, 'class_edible']].groupby([feature]).count().sort_values(by='class_edible', ascending=False))
    # Quale è la ratio fra una feature e la sua velenosità?
    print(dati[[feature, 'class_edible']].groupby([feature]).mean().sort_values(by='class_edible', ascending=False))

### TEST e TRAIN ###
y = dati["class_edible"].values
# Mi analizzo tutti le feature ad eccezione di class_edible
x = dati.drop(["class_edible"], axis=1).values
# Uso come dimensione del test, metà del file, se dovesse scaturirne una accuratezza bassa, è da aumentare la test_size
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)

# Come solver uso quello di default, e come iterazione massima per la convergenza 1000 (Anche se non è l'ideale in
# termine di prestazioni)
logistic_reg = LogisticRegression(max_iter=1000)
logistic_reg.fit(x_train, y_train)
print("\nAccuracy: ", (logistic_reg.score(x_test, y_test) * 100), "%")
