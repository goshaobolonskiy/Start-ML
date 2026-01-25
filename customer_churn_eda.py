import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
print("Размер данных:", df.shape)
print("\nПервые 5 строк:")
print(df.head())
print("\nИнформация о данных:")
print(df.info())
print("\nПропуски:")
print(df.isnull().sum())
print("\nОписательная статистика:")
print(df.describe())

# Перевод столбца в тип данных float с учетом ошибок
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# Удаление ненужных данных
a = ["customerID"]
df_test = df.drop(a, axis=1)

# Корреляции численных столбцов
print('Корреляции численных столбцов')
corr_numbers = df_test.select_dtypes(include=['float', 'int']).corr()
corr_triu = corr_numbers.where(np.triu(np.ones(corr_numbers.shape), k=1).astype(bool))
corr_pairs = corr_triu.stack()
for (f1, f2), value in corr_pairs.sort_values(ascending=False).items():
    print(f"  {f1:15} ↔ {f2:15}: {value:+.4f}")

# Распределение каждой из переменной
for i in df_test.columns:
    plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    plt.title(f"Распределение {i}")
    plt.xlabel(f"Значение {i}")
    plt.ylabel("Количество")
    plt.hist(df_test[i])
    plt.show()

# Список категориальных столбцов, которые будем кодировать
cat = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn']
cat = [column for column in df_test.columns if df_test[column].dtype == 'O']
# Кодирование каждого категориального значения цифрой
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(drop="if_binary", sparse_output=False)

new_category_columns = ohe.fit_transform(df_test[cat])

df_test = pd.concat([
    df_test.drop(columns=cat),
    pd.DataFrame(new_category_columns, columns=ohe.get_feature_names_out())
], axis=1)

# Вывод только категориальных столбоцов
# print(df.select_dtypes(include=['category', 'object']).info())


# Матрица корреляции с целевой переменной
corr_with_target = df_test.corr()[["Churn_Yes"]]
corr_with_target = corr_with_target['Churn_Yes'].drop('Churn_Yes')
print('Корреляция с целевой переменной')
print(corr_with_target)
# Матрица корреляции
corr_matrix = df_test.corr()

# Визуализация матрицы корреляций
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.show()

# Самые большие корреляции с целевой переменной
top_features = corr_with_target.abs().sort_values(ascending=False).head(5).index
top_correlations = corr_with_target.loc[top_features].sort_values(ascending=False)
print(f"Топ-5 самых сильных корреляций с целевой переменной (с сохранением знака):\n{top_correlations}")

# plt.title('Корреляция деятельности с оценкой')
# plt.xlabel('Признаки')
# plt.ylabel('Churn')

# Гистограмма корреляций
bars = plt.bar(top_correlations.index, top_correlations.values, color='skyblue', width=0.8, edgecolor='black',
               alpha=0.7)
plt.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
plt.show()
