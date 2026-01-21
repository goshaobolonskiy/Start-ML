import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Загрузка данных
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# Удаление ненужных данных
a = ["customerID"]
df = df.drop(a, axis=1)

# Пример кода для анализа и удаления численных выбросов
numb_feature = df.select_dtypes(include=['float', 'int'])  # df содержащий только переменные с численными данными
for i in numb_feature:
    Q1 = df[i].quantile(0.25)  # 1й квантиль = 25 перценталь
    Q3 = df[i].quantile(0.75)  # 3й квантиль = 75 перценталь
    IQR = Q3 - Q1  # Среднее значение (Q2) - медиана
    lower_bound = Q1 - 1.5 * IQR  # Вычисление нижней границе по формуле
    upper_bound = Q3 + 1.5 * IQR  # Вычисление верхней границы
    outliers_mask = (df[i] < lower_bound) | (
            df[i] > upper_bound)  # df с true/false, в зависимости попадает ли в нижнюю и верхнюю границы
    df = df[~outliers_mask]  # Удаление строчек с true
    # print(f'Границы для {i}: [{lower_bound}, {upper_bound}]')


# Обнаружение выбросов в категориальных признаках
def detect_rare_categories(df, column, threshold=0.01):
    """
    Обнаружение редких категорий по порогу частоты
    """
    freq = df[column].value_counts(normalize=True)
    rare_cats = freq[freq < threshold].index.tolist()

    return rare_cats, freq


cat_feature = df.select_dtypes(include=['category', 'object'])
# Использование функции
for i in cat_feature:
    rare_categories, frequencies = detect_rare_categories(df, i, threshold=0.05)
    if rare_categories == []:
        continue
    else:
        print(f"В {i} найдена редкая категория {rare_categories}")

# Кодирование категориальных данных
cat = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn']
# Кодирование каждого категориального значения цифрой
le = LabelEncoder()
for i in cat:
    df[f'{i}'] = le.fit_transform(df[f'{i}'])
