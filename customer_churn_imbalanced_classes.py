from customer_churn_outliers import *

# print(f'Распределение целевой переменной равно:')
# print(df["Churn"].value_counts(normalize=True))

# X - все признаки, кроме целевой переменной
X = df.drop('Churn', axis=1)  # axis=1 означает удаление столбца

# y - только целевая переменная
y = df['Churn']

# Удаление пропусков
X = X.dropna()
y = y[X.index]

# Рандомный UnderSampling
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(
    sampling_strategy='auto',  # 'auto' для баланса 1:1, float для заданного соотношения
    random_state=42,
    replacement=False  # Без повторений
)
X_resampled, y_resampled = rus.fit_resample(X, y)
# print(f"X: {len(X_resampled)}, y: {len(y_resampled)}")
# print(y.value_counts(), y_resampled.value_counts())


# Удаление с помощью TomikLinks
from imblearn.under_sampling import TomekLinks

tl = TomekLinks(
    sampling_strategy='auto'  # или 'majority', 'not minority', 'all'
)
X_resampled, y_resampled = tl.fit_resample(X, y)
# print(f"X: {len(X_resampled)}, y: {len(y_resampled)}")
# print(y.value_counts(), y_resampled.value_counts())


# Удаляет примеры, которые отличаются от класса большинства его k ближайших соседей
from imblearn.under_sampling import EditedNearestNeighbours

enn = EditedNearestNeighbours(
    sampling_strategy='auto',
    n_neighbors=3,
    kind_sel='all'  # 'all' или 'mode'
)
X_resampled, y_resampled = enn.fit_resample(X, y)
# print(y.value_counts(), y_resampled.value_counts())


# Синтетическое добавление данных путем линейной интерполяции
from imblearn.over_sampling import SMOTE

smote = SMOTE(
    sampling_strategy='auto',  # или float (0.5 = 50% миноритарного класса)
    k_neighbors=5,
    random_state=42
)
X_resampled, y_resampled = smote.fit_resample(X, y)
# print(y.value_counts(), y_resampled.value_counts())


# Фокусируется на примерах миноритарного класса,находящихся на границе с мажоритарным классом
from imblearn.over_sampling import BorderlineSMOTE

border_smote = BorderlineSMOTE(
    sampling_strategy='auto',
    k_neighbors=5,
    m_neighbors=10,
    kind='borderline-1',  # или 'borderline-2'
    random_state=42
)
X_resampled, y_resampled = border_smote.fit_resample(X, y)
# print(y.value_counts(), y_resampled.value_counts())


# Создает больше синтетических примеров в областях, где миноритарные примеры трудно классифицировать.
from imblearn.over_sampling import ADASYN

adasyn = ADASYN(
    sampling_strategy='auto',
    n_neighbors=5,
    random_state=42
)
X_resampled, y_resampled = adasyn.fit_resample(X, y)
# print(y.value_counts(), y_resampled.value_counts())


# Комбинация из SMOTE и ENN
# SMOTE для увеличения миноритарного класса, затем ENN для очистки данных от шума.
from imblearn.combine import SMOTEENN

smote_enn = SMOTEENN(
    sampling_strategy='auto',
    smote=SMOTE(k_neighbors=3),
    enn=EditedNearestNeighbours(n_neighbors=3),
    random_state=42
)
X_resampled, y_resampled = smote_enn.fit_resample(X, y)
# print(y.value_counts(), y_resampled.value_counts())


# Комбинация из SMOTE и TomikLinks
from imblearn.combine import SMOTETomek

smote_tomek = SMOTETomek(
    sampling_strategy='auto',
    smote=SMOTE(k_neighbors=5),
    tomek=TomekLinks(),
    random_state=42
)
X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
# print(y.value_counts(), y_resampled.value_counts())
