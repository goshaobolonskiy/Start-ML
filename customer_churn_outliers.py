import pandas as pd
import numpy as np
from customer_churn_Split import *


# Обнаружение численных выбросов

# Обнаружение с помощью Box plot
def detection_box_plot(X_train):
    # 1. Вычисляем границы ТОЛЬКО на исходном X_train (до удаления)
    numeric_cols = X_train.select_dtypes(include=['float', 'int']).columns

    # Словари для хранения границ
    lower_bounds = {}
    upper_bounds = {}

    # Маска для всех выбросов
    all_outliers_mask = pd.Series(False, index=X_train.index)

    for col in numeric_cols:
        Q1 = X_train[col].quantile(0.25)
        Q3 = X_train[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bounds[col] = Q1 - 1.5 * IQR
        upper_bounds[col] = Q3 + 1.5 * IQR

        # Добавляем выбросы этого признака в общую маску
        col_outliers = (X_train[col] < lower_bounds[col]) | (X_train[col] > upper_bounds[col])
        all_outliers_mask = all_outliers_mask | col_outliers

    return all_outliers_mask, lower_bounds, upper_bounds


# Обнаружение с помощью MAD (Median Absolute Deviation)
def detection_MAD(X_train, k=3.0):
    # 1. Вычисляем границы ТОЛЬКО на исходном X_train
    numeric_cols = X_train.select_dtypes(include=['float', 'int']).columns

    # Словари для хранения границ
    lower_bounds = {}
    upper_bounds = {}

    # Маска для всех выбросов
    all_outliers_mask = pd.Series(False, index=X_train.index)

    for col in numeric_cols:
        median = X_train[col].median()  # Вычисляем медиану

        mad = (X_train[col] - median).abs().median()  # Вычисляем MAD: медиана абсолютных отклонений от медианы

        # Для нормального распределения: 1.4826 * MAD ≈ σ
        # Но часто используют просто MAD для определения выбросов
        lower_bounds[col] = median - k * mad
        upper_bounds[col] = median + k * mad

        # Добавляем выбросы этого признака в общую маску
        col_outliers = (X_train[col] < lower_bounds[col]) | (X_train[col] > upper_bounds[col])
        all_outliers_mask = all_outliers_mask | col_outliers

    return all_outliers_mask, lower_bounds, upper_bounds


# Обнаружение с помощью KKN (не работает с nan значениями)
def detection_knn(X_train, n_neighbors=5, contamination=0.05, method='distance'):
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    """
    Обнаружение выбросов методом k-ближайших соседей (KNN)

    Parameters:
    -----------
    X_train : DataFrame
        Тренировочные данные
    n_neighbors : int, default=5
        Количество соседей для KNN
    contamination : float, default=0.05
        Доля ожидаемых выбросов (0.0-0.5)
    method : str, default='distance'
        Метод обнаружения: 'distance', 'mean_distance', 'std_distance'

    Returns:
    --------
    outliers_mask : Series
        Маска выбросов (True - выброс, False - нормальное значение)
    distances_info : dict
        Словарь с дополнительной информацией о расстояниях
    """
    # 1. Масштабируем данные (KNN чувствителен к масштабу)
    numeric_cols = X_train.select_dtypes(include=['float', 'int']).columns
    X_numeric = X_train[numeric_cols].copy()

    if X_numeric.isna().any().any():
        print("Внимание: DBSCAN не работает с пропущенными значениями.")
        print("Рекомендуется предварительно обработать пропуски.")
        # Создаем маску без выбросов
        return pd.Series(False, index=X_train.index), None

    # Проверяем, есть ли достаточно данных
    if len(X_numeric) <= n_neighbors:
        print(f"Предупреждение: слишком мало данных ({len(X_numeric)}) для n_neighbors={n_neighbors}")
        n_neighbors = max(1, len(X_numeric) - 1)

    # 2. Масштабирование (стандартизация)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_numeric)

    # 3. Находим k ближайших соседей
    knn = NearestNeighbors(n_neighbors=n_neighbors + 1)  # +1 чтобы исключить саму точку
    knn.fit(X_scaled)

    # Расстояния до соседей (первый столбец - расстояние до самой себя = 0)
    distances, indices = knn.kneighbors(X_scaled)

    # Исключаем расстояние до самой себя (первый столбец)
    neighbor_distances = distances[:, 1:]

    # 4. Вычисляем метрики в зависимости от метода
    if method == 'distance':
        # Расстояние до k-го соседа
        metric_values = neighbor_distances[:, -1]  # последний (k-й) сосед
        threshold = np.percentile(metric_values, (1 - contamination) * 100)
        outliers_mask = metric_values > threshold

    elif method == 'mean_distance':
        # Среднее расстояние до всех k соседей
        metric_values = neighbor_distances.mean(axis=1)
        threshold = np.percentile(metric_values, (1 - contamination) * 100)
        outliers_mask = metric_values > threshold

    elif method == 'std_distance':
        # Стандартное отклонение расстояний до соседей
        metric_values = neighbor_distances.std(axis=1)
        threshold = np.percentile(metric_values, (1 - contamination) * 100)
        outliers_mask = metric_values > threshold

    else:
        raise ValueError(f"Неизвестный метод: {method}. Используйте 'distance', 'mean_distance' или 'std_distance'")

    # 5. Создаем маску с правильными индексами
    outliers_mask_series = pd.Series(outliers_mask, index=X_train.index)

    # 6. Дополнительная информация
    distances_info = {
        'metric_values': metric_values,
        'threshold': threshold,
        'scaler': scaler,
        'knn': knn,
        'method': method,
        'contamination': contamination,
        'n_neighbors': n_neighbors
    }

    return outliers_mask_series, distances_info


# Обнаружение с помощью DBScan (не работает с nan значениями)
def detection_dbscan(X_train, eps=0.5, min_samples=5, auto_eps=True, scale_data=True):
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    """
    Обнаружение выбросов методом DBSCAN

    Parameters:
    -----------
    X_train : DataFrame
        Тренировочные данные
    eps : float, default=0.5
        Максимальное расстояние между точками одного кластера
    min_samples : int, default=5
        Минимальное количество точек для формирования кластера
    auto_eps : bool, default=True
        Автоматический подбор eps на основе k-расстояний
    scale_data : bool, default=True
        Масштабировать ли данные перед DBSCAN

    Returns:
    --------
    outliers_mask : Series
        Маска выбросов (True - выброс/шум, False - нормальная точка)
    dbscan_info : dict
        Словарь с дополнительной информацией
    """
    # 1. Выбираем только числовые признаки
    numeric_cols = X_train.select_dtypes(include=['float', 'int']).columns
    X_numeric = X_train[numeric_cols].copy()

    # Проверяем наличие пропусков
    if X_numeric.isna().any().any():
        print("Внимание: DBSCAN не работает с пропущенными значениями.")
        print("Рекомендуется предварительно обработать пропуски.")
        # Создаем маску без выбросов
        return pd.Series(False, index=X_train.index), None

    # 2. Масштабируем данные (DBSCAN чувствителен к масштабу)
    if scale_data:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_numeric)
    else:
        scaler = None
        X_scaled = X_numeric.values

    # 3. Автоматический подбор eps (метод локтя по k-расстояниям)
    if auto_eps and len(X_scaled) > min_samples:
        try:
            from sklearn.neighbors import NearestNeighbors
            # Вычисляем расстояния до k-го соседа (k = min_samples)
            neighbors = NearestNeighbors(n_neighbors=min_samples)
            neighbors_fit = neighbors.fit(X_scaled)
            distances, indices = neighbors_fit.kneighbors(X_scaled)

            # Сортируем расстояния до k-го соседа
            k_distances = np.sort(distances[:, min_samples - 1])

            # Находим "локоть" графика (точку наибольшей кривизны)
            # Простой метод - используем вторую производную
            if len(k_distances) > 10:
                # Берем производные
                x = np.arange(len(k_distances))
                first_deriv = np.gradient(k_distances)
                second_deriv = np.gradient(first_deriv)

                # Находим точку с максимальной второй производной
                # (резкое изменение наклона)
                elbow_idx = np.argmax(second_deriv)

                # Устанавливаем eps как расстояние в этой точке
                auto_eps_value = k_distances[elbow_idx]

                # Ограничиваем eps разумными пределами
                auto_eps_value = max(0.1, min(auto_eps_value, 5.0))

                print(f"Автоматический подбор eps: {auto_eps_value:.3f}")
                eps = auto_eps_value

                k_dist_info = {
                    'distances': k_distances,
                    'elbow_idx': elbow_idx,
                    'auto_eps': auto_eps_value
                }
            else:
                k_dist_info = None
        except:
            k_dist_info = None
            print("Не удалось автоматически подобрать eps, используется значение по умолчанию")
    else:
        k_dist_info = None

    # 4. Применяем DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    labels = dbscan.fit_predict(X_scaled)

    # 5. Выбросы - это точки с меткой -1 (шум)
    outliers_mask_array = labels == -1

    # 6. Создаем маску с правильными индексами
    outliers_mask = pd.Series(outliers_mask_array, index=X_train.index)

    # 7. Анализируем результаты
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # исключаем шум
    n_noise = np.sum(labels == -1)
    n_core_samples = np.sum(dbscan.core_sample_indices_ != -1) if hasattr(dbscan, 'core_sample_indices_') else 0

    # 8. Собираем информацию
    dbscan_info = {
        'labels': labels,
        'eps': eps,
        'min_samples': min_samples,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'n_core_samples': n_core_samples,
        'noise_percentage': (n_noise / len(labels)) * 100 if len(labels) > 0 else 0,
        'scaler': scaler,
        'dbscan': dbscan,
        'k_dist_info': k_dist_info
    }

    return outliers_mask, dbscan_info


# Обработка численных выбросов

# Удаление численных выбросов (только для тренировочных данных)
def deleting_outliers_train(X_train, all_outliers_mask):
    X_train_clean = X_train[~all_outliers_mask].copy()

    return X_train_clean


# Каппинг - только для тестовых данных
def capping_outliers_test(X_train, X_test, lower_bounds, upper_bounds, exclude_columns=None):
    """
    Capping с возможностью исключения определенных колонок
    """
    if exclude_columns is None:
        exclude_columns = []

    X_test_capped = X_test.copy()

    # Находим общие числовые колонки, исключая указанные
    numeric_cols = X_train.select_dtypes(include=['float', 'int']).columns.intersection(
        X_test.select_dtypes(include=['float', 'int']).columns
    )

    # Удаляем исключенные колонки
    numeric_cols = [col for col in numeric_cols if col not in exclude_columns]

    for col in numeric_cols:
        # Проверяем, что границы существуют для этой колонки
        if col in lower_bounds and col in upper_bounds:
            X_test_capped[col] = X_test[col].clip(
                lower=lower_bounds[col],
                upper=upper_bounds[col]
            )

    return X_test_capped


# К обработке выбросов не относится, нормализует распределение
def transform_yeo_johnson(X_train, X_test, features=None, standardize=True):
    from sklearn.preprocessing import PowerTransformer
    """
    Применяет Yeo-Johnson преобразование к данным

    Parameters:
    -----------
    X_train, X_test : DataFrame
        Тренировочные и тестовые данные
    features : list или None
        Список признаков для преобразования (если None - все числовые)
    standardize : bool, default=True
        Стандартизировать ли результат (mean=0, std=1)

    Returns:
    --------
    X_train_transformed, X_test_transformed : DataFrames
        Преобразованные данные
    yeo_transformer : обученный PowerTransformer
    """
    # 1. Создаем копии данных
    X_train_transformed = X_train.copy()
    X_test_transformed = X_test.copy()

    # 2. Определяем признаки для преобразования
    if features is None:
        features = X_train.select_dtypes(include=['float', 'int']).columns.tolist()

    # 3. Проверяем, что признаки существуют в обоих наборах
    common_features = [f for f in features if f in X_train.columns and f in X_test.columns]

    if len(common_features) == 0:
        print("Нет общих признаков для преобразования")
        return X_train_transformed, X_test_transformed, None

    # 4. Создаем и трансформер
    yeo_transformer = PowerTransformer(method='yeo-johnson', standardize=standardize)

    # Обучаем на тренировочных данных
    X_train_transformed[common_features] = yeo_transformer.fit_transform(
        X_train[common_features]
    )

    # Применяем к тестовым данным (используем параметры с тренировочных)
    X_test_transformed[common_features] = yeo_transformer.transform(
        X_test[common_features]
    )

    # 5. Возвращаем информацию о преобразовании
    print(f"Yeo-Johnson применен к {len(common_features)} признакам")
    print(f"Найденные λ (lambda) параметры:")
    for feature, lambda_val in zip(common_features, yeo_transformer.lambdas_):
        print(f"  {feature}: λ = {lambda_val:.3f}")

    return X_train_transformed, X_test_transformed, yeo_transformer


# Обработка аномалии за счет добавления колонки была ли аномалия
def add_outlier_flags(X_train, X_test, lower_bounds, upper_bounds, prefix='outlier_'):
    """
    Добавляет бинарные признаки о выбросах
    """
    # Копируем данные
    X_train_with_flags = X_train.copy()
    X_test_with_flags = X_test.copy()

    # Обнаруживаем выбросы на тренировочных данных

    # Для каждого числового признака создаем флаг
    numeric_cols = X_train.select_dtypes(include=['float', 'int']).columns

    for col in numeric_cols:
        if col in lower_bounds:
            # Флаг для тренировочных данных
            train_flag = (X_train[col] < lower_bounds[col]) | (X_train[col] > upper_bounds[col])
            X_train_with_flags[f'{prefix}{col}'] = train_flag.astype(int)

            # Флаг для тестовых данных (по тем же границам!)
            test_flag = (X_test[col] < lower_bounds[col]) | (X_test[col] > upper_bounds[col])
            X_test_with_flags[f'{prefix}{col}'] = test_flag.astype(int)

    return X_train_with_flags, X_test_with_flags


# all_outliers_mask, lower_bounds, upper_bounds = detection_box_plot(X_train)
# print(capping_outliers(X_train, X_test, lower_bounds, upper_bounds))
# print(deleting_outliers(X_train, y_train, all_outliers_mask))
# a, b = add_outlier_flags(X_train, X_test, lower_bounds, upper_bounds)
# print(a.info())


# Обнаружение выбросов в категориальных признаках

# Обнаружение редких значений
def detection_rare_categories(X_train, threshold=0.01, min_count=5):
    """
    Обнаружение редких категорий в категориальных признаках

    Parameters:
    -----------
    X_train : DataFrame
        Тренировочные данные
    threshold : float, default=0.01
        Порог относительной частоты (1%)
    min_count : int, default=5
        Минимальное абсолютное количество

    Returns:
    --------
    outliers_mask : Series
        Маска строк, содержащих редкие категории
    rare_info : dict
        Словарь с информацией о редких категориях
    """
    # Создаем пустую маску
    outliers_mask = pd.Series(False, index=X_train.index)

    # Словарь для хранения информации
    rare_info = {}

    # Выбираем категориальные признаки
    categorical_cols = X_train.select_dtypes(
        include=['object', 'category', 'string']
    ).columns

    for col in categorical_cols:
        # Считаем частоты
        value_counts = X_train[col].value_counts()
        total = len(X_train[col])

        # Находим редкие категории по двум критериям
        rare_by_freq = value_counts[value_counts / total < threshold].index
        rare_by_count = value_counts[value_counts < min_count].index

        # Объединяем
        rare_categories = set(rare_by_freq) | set(rare_by_count)

        if len(rare_categories) > 0:
            # Добавляем в информацию
            rare_info[col] = {
                'categories': list(rare_categories),
                'counts': value_counts.loc[rare_categories].to_dict(),
                'percentage': (value_counts.loc[rare_categories].sum() / total * 100)
            }

            # Обновляем маску
            col_mask = X_train[col].isin(rare_categories)
            outliers_mask = outliers_mask | col_mask

    return outliers_mask, rare_info


# Обнаружение строковых аномалий
def detection_string_anomalies(X_train, min_length=1, max_length=50,
                               allow_digits=False, allow_special_chars=True):
    """
    Обнаружение аномальных строковых значений

    Parameters:
    -----------
    X_train : DataFrame
    min_length : int, минимальная допустимая длина
    max_length : int, максимальная допустимая длина
    allow_digits : bool, разрешены ли цифры в строке
    allow_special_chars : bool, разрешены ли специальные символы

    Returns:
    --------
    outliers_mask : Series
    anomalies_info : dict
    """
    outliers_mask = pd.Series(False, index=X_train.index)
    anomalies_info = {}

    categorical_cols = X_train.select_dtypes(
        include=['object', 'category', 'string']
    ).columns

    for col in categorical_cols:
        col_info = {
            'length_outliers': [],
            'digit_outliers': [],
            'special_char_outliers': []
        }

        for idx, value in X_train[col].items():
            if pd.isna(value):
                continue

            value_str = str(value)

            # Проверка длины
            if len(value_str) < min_length or len(value_str) > max_length:
                col_info['length_outliers'].append((idx, value_str))
                outliers_mask[idx] = True

            # Проверка цифр (если не разрешены)
            if not allow_digits and any(char.isdigit() for char in value_str):
                col_info['digit_outliers'].append((idx, value_str))
                outliers_mask[idx] = True

            # Проверка специальных символов (если не разрешены)
            if not allow_special_chars:
                import re
                # Допустим, только буквы, цифры и пробелы
                if re.search(r'[^\w\s]', value_str):
                    col_info['special_char_outliers'].append((idx, value_str))
                    outliers_mask[idx] = True

        # Сохраняем информацию только если есть аномалии
        if any(col_info.values()):
            anomalies_info[col] = col_info

    return outliers_mask, anomalies_info


# Обнаружение на основе неожиданных значений (желательно иметь белый список)
def detection_unexpected_values(X_train, whitelist_dict=None):
    """
    Обнаружение категорий, не входящих в белый список

    Parameters:
    -----------
    X_train : DataFrame
    whitelist_dict : dict
        Словарь {признак: [допустимые_значения]}
        Если None, создается на основе частых значений

    Returns:
    --------
    outliers_mask : Series
    unexpected_info : dict
    """
    outliers_mask = pd.Series(False, index=X_train.index)
    unexpected_info = {}

    categorical_cols = X_train.select_dtypes(
        include=['object', 'category', 'string']
    ).columns

    # Если белый список не предоставлен, создаем на основе топ-N значений
    if whitelist_dict is None:
        whitelist_dict = {}
        for col in categorical_cols:
            top_values = X_train[col].value_counts().head(20).index.tolist()
            whitelist_dict[col] = top_values

    for col in categorical_cols:
        if col not in whitelist_dict:
            continue

        allowed_values = set(whitelist_dict[col])
        unique_values = set(X_train[col].dropna().unique())

        # Находим неожиданные значения
        unexpected = unique_values - allowed_values

        if len(unexpected) > 0:
            # Маска для текущего признака
            col_mask = X_train[col].isin(unexpected)
            outliers_mask = outliers_mask | col_mask

            # Сохраняем информацию
            unexpected_info[col] = {
                'unexpected_values': list(unexpected),
                'allowed_values': list(allowed_values),
                'counts': X_train[col].value_counts().loc[list(unexpected)].to_dict()
            }

    return outliers_mask, unexpected_info


# Объединяющая функция, которая ищет все сразу
def detection_categorical_outliers(X_train, methods=None, **kwargs):
    """
    Основная функция для обнаружения категориальных выбросов

    Parameters:
    -----------
    X_train : DataFrame
    methods : list или None
        Список методов для применения:
        - 'rare': редкие категории
        - 'string': строковые аномалии
        - 'unexpected': неожиданные значения
        Если None, применяются все методы

    Returns:
    --------
    all_outliers_mask : Series
    combined_info : dict
    """
    if methods is None:
        methods = ['rare', 'string', 'unexpected']

    all_outliers_mask = pd.Series(False, index=X_train.index)
    combined_info = {}

    # Применяем выбранные методы
    if 'rare' in methods:
        rare_mask, rare_info = detection_rare_categories(
            X_train,
            threshold=kwargs.get('threshold', 0.01),
            min_count=kwargs.get('min_count', 5)
        )
        all_outliers_mask = all_outliers_mask | rare_mask
        combined_info['rare_categories'] = rare_info

    if 'string' in methods:
        string_mask, string_info = detection_string_anomalies(
            X_train,
            min_length=kwargs.get('min_length', 1),
            max_length=kwargs.get('max_length', 100),
            allow_digits=kwargs.get('allow_digits', False),
            allow_special_chars=kwargs.get('allow_special_chars', True)
        )
        all_outliers_mask = all_outliers_mask | string_mask
        combined_info['string_anomalies'] = string_info

    if 'unexpected' in methods:
        unexpected_mask, unexpected_info = detection_unexpected_values(
            X_train,
            whitelist_dict=kwargs.get('whitelist_dict', None)
        )
        all_outliers_mask = all_outliers_mask | unexpected_mask
        combined_info['unexpected_values'] = unexpected_info

    # Статистика
    combined_info['summary'] = {
        'total_outliers': all_outliers_mask.sum(),
        'percentage': all_outliers_mask.mean() * 100,
        'methods_applied': methods
    }

    return all_outliers_mask, combined_info


# Обработка категориальных выбросов

# Группировка редких категорий в одну
def group_rare_categories(X_train, X_test, detection_info=None, threshold=0.01, replacement='Other'):
    """
    Группирует редкие категории в указанную замену

    Parameters:
    -----------
    X_train, X_test : DataFrame
    detection_info : dict (результат detection_categorical_outliers)
        Если None, будет вычислен автоматически
    threshold : float, порог для редких категорий
    replacement : str, значение для замены редких категорий

    Returns:
    --------
    X_train_processed, X_test_processed : DataFrames
    grouping_rules : dict, правила группировки
    """
    # Если detection_info не передан, вычисляем его
    if detection_info is None:
        _, detection_info = detection_categorical_outliers(
            X_train, methods=['rare'], threshold=threshold
        )

    # Копируем данные
    X_train_processed = X_train.copy()
    X_test_processed = X_test.copy()

    # Словарь правил группировки
    grouping_rules = {}

    # Получаем информацию о редких категориях
    rare_info = detection_info.get('rare_categories', {})

    for col, info in rare_info.items():
        if col not in X_train_processed.columns:
            continue

        # Получаем редкие категории для этого признака
        rare_categories = set(info['categories'])

        # На всякий случай переводим все в нижний регистр:
        rare_categories_lower = {cat.lower() for cat in rare_categories}

        # Применяем группировку к train
        X_train_processed[col] = X_train_processed[col].apply(  # Применяет к каждому элементу колонки
            # Если содержится в множестве редких, то заменяем
            lambda x: replacement if str(x).lower() in rare_categories_lower else x
        )

        # Применяем те же правила к test
        if col in X_test_processed.columns:
            X_test_processed[col] = X_test_processed[col].apply(
                lambda x: replacement if x in rare_categories else x
            )

        # Сохраняем правило
        grouping_rules[col] = {
            'rare_categories': list(rare_categories),
            'replacement': replacement,
            'count_affected_train': (X_train[col].isin(rare_categories)).sum(),
            'count_affected_test': (X_test[col].isin(rare_categories)).sum() if col in X_test.columns else 0
        }

    return X_train_processed, X_test_processed, grouping_rules


# Заменя на моду (наиболее частую категорию)
def replace_with_mode(X_train, X_test, detection_info=None):
    """
    Заменяет выбросы на моду (наиболее частую категорию)

    Parameters:
    -----------
    X_train, X_test : DataFrame
    detection_info : dict (результат detection_categorical_outliers)

    Returns:
    --------
    X_train_processed, X_test_processed : DataFrames
    replacement_rules : dict, правила замены
    """
    # Если detection_info не передан, вычисляем его
    if detection_info is None:
        _, detection_info = detection_categorical_outliers(X_train)

    # Копируем данные
    X_train_processed = X_train.copy()
    X_test_processed = X_test.copy()

    # Словарь правил замены
    replacement_rules = {}

    # Обрабатываем каждый тип выбросов
    for method_name, method_info in detection_info.items():
        if method_name == 'summary':
            continue

        for col, info in method_info.items():
            if col not in X_train_processed.columns:
                continue

            # Определяем категории для замены
            categories_to_replace = set()

            if 'categories' in info:  # для редких категорий
                categories_to_replace.update(info['categories'])
            elif 'unexpected_values' in info:  # для неожиданных значений
                categories_to_replace.update(info['unexpected_values'])

            # Добавляем строковые аномалии
            if 'string_anomalies' in detection_info:
                str_info = detection_info['string_anomalies'].get(col, {})
                for anomaly_type in ['length_outliers', 'digit_outliers', 'special_char_outliers']:
                    if anomaly_type in str_info:
                        for idx, val in str_info[anomaly_type]:
                            categories_to_replace.add(val)

            if not categories_to_replace:
                continue

            # Вычисляем моду (наиболее частую категорию) для этого признака
            # Исключаем выбросы при вычислении моды
            valid_values = X_train_processed[~X_train_processed[col].isin(categories_to_replace)][col]
            if len(valid_values) > 0:
                mode_value = valid_values.mode()[0] if not valid_values.mode().empty else replacement
            else:
                mode_value = replacement

            # Заменяем в train
            mask_train = X_train_processed[col].isin(categories_to_replace)
            X_train_processed.loc[mask_train, col] = mode_value

            # Заменяем в test
            if col in X_test_processed.columns:
                mask_test = X_test_processed[col].isin(categories_to_replace)
                X_test_processed.loc[mask_test, col] = mode_value

            # Сохраняем правило
            replacement_rules[col] = {
                'categories_to_replace': list(categories_to_replace),
                'replacement_value': mode_value,
                'count_replaced_train': mask_train.sum(),
                'count_replaced_test': mask_test.sum() if col in X_test.columns else 0
            }

    return X_train_processed, X_test_processed, replacement_rules


# Создание бинарных признаков
def add_outlier_flags(X_train, X_test, detection_info=None, prefix='outlier_'):
    """
    Создает бинарные признаки, указывающие на наличие выброса

    Parameters:
    -----------
    X_train, X_test : DataFrame
    detection_info : dict (результат detection_categorical_outliers)
    prefix : str, префикс для названий новых признаков

    Returns:
    --------
    X_train_with_flags, X_test_with_flags : DataFrames с добавленными признаками
    flag_info : dict, информация о созданных флагах
    """
    # Если detection_info не передан, вычисляем его
    if detection_info is None:
        _, detection_info = detection_categorical_outliers(X_train)

    # Копируем данные
    X_train_with_flags = X_train.copy()
    X_test_with_flags = X_test.copy()

    # Словарь информации о флагах
    flag_info = {}

    # Обрабатываем каждый тип выбросов
    for method_name, method_info in detection_info.items():
        if method_name == 'summary':
            continue

        for col, info in method_info.items():
            if col not in X_train.columns:
                continue

            # Определяем категории, считающиеся выбросами
            outlier_categories = set()

            if 'categories' in info:  # редкие категории
                outlier_categories.update(info['categories'])
            elif 'unexpected_values' in info:  # неожиданные значения
                outlier_categories.update(info['unexpected_values'])

            # Добавляем строковые аномалии
            if 'string_anomalies' in detection_info:
                str_info = detection_info['string_anomalies'].get(col, {})
                for anomaly_type in ['length_outliers', 'digit_outliers', 'special_char_outliers']:
                    if anomaly_type in str_info:
                        for idx, val in str_info[anomaly_type]:
                            outlier_categories.add(val)

            if not outlier_categories:
                continue

            # Создаем бинарный признак для train
            flag_name = f"{prefix}{col}_{method_name}"
            X_train_with_flags[flag_name] = X_train[col].isin(outlier_categories).astype(int)

            # Создаем бинарный признак для test
            if col in X_test.columns:
                X_test_with_flags[flag_name] = X_test[col].isin(outlier_categories).astype(int)

            # Сохраняем информацию
            flag_info[flag_name] = {
                'original_column': col,
                'method': method_name,
                'outlier_categories': list(outlier_categories),
                'count_outliers_train': X_train_with_flags[flag_name].sum(),
                'count_outliers_test': X_test_with_flags[
                    flag_name].sum() if flag_name in X_test_with_flags.columns else 0
            }

    return X_train_with_flags, X_test_with_flags, flag_info


# Исправление опечаток
def correct_typos(X_train, X_test, correction_dict=None, fuzzy_threshold=0.8,
                  normalize=True, use_multiple_metrics=False):
    """
    Исправляет опечатки в категориальных признаках с улучшениями

    Parameters:
    -----------
    X_train, X_test : DataFrame
    correction_dict : dict
        Словарь исправлений {признак: {неправильное: правильное}}
    fuzzy_threshold : float
        Порог для fuzzy matching (0-1)
    normalize : bool
        Нормализовать ли строки (к нижнему регистру, удаление лишних пробелов)
    use_multiple_metrics : bool
        Использовать ли несколько метрик сравнения

    Returns:
    --------
    X_train_corrected, X_test_corrected : DataFrames
    correction_info : dict, информация об исправлениях
    """
    import pandas as pd
    import numpy as np

    # Копируем данные
    X_train_corrected = X_train.copy()
    X_test_corrected = X_test.copy()

    # Словарь исправлений
    if correction_dict is None:
        correction_dict = {}

    # Словарь для информации об исправлениях
    correction_info = {}

    # Категориальные колонки
    categorical_cols = X_train.select_dtypes(
        include=['object', 'category', 'string']
    ).columns

    # Функция нормализации строк
    def normalize_string(s):
        if pd.isna(s):
            return s
        # Приводим к строке, на случай если это не строка
        s = str(s)
        # Удаляем лишние пробелы
        s = ' '.join(s.split())
        # Приводим к нижнему регистру
        s = s.lower()
        return s

    # Автоматическое обнаружение опечаток (если не заданы исправления)
    if not correction_dict:
        try:
            from rapidfuzz import process, fuzz
            use_fuzzy = True
        except ImportError:
            print("Библиотека rapidfuzz не установлена. Установите: pip install rapidfuzz")
            use_fuzzy = False
            return X_train_corrected, X_test_corrected, {}

        for col in categorical_cols:
            # Получаем уникальные значения
            unique_values = X_train[col].dropna().unique()

            if len(unique_values) <= 1:
                continue

            # Создаем словарь исправлений для этой колонки
            col_corrections = {}

            # Подготавливаем данные для сравнения
            # Создаем список кортежей (оригинальное_значение, нормализованное_значение)
            values_to_compare = []
            for value in unique_values:
                if pd.isna(value):
                    continue
                orig_value = str(value)
                norm_value = normalize_string(orig_value) if normalize else orig_value
                values_to_compare.append((orig_value, norm_value))

            # Для каждого значения ищем похожие
            for i, (orig_value, norm_value) in enumerate(values_to_compare):
                # Создаем список для сравнения (без текущего элемента)
                other_values = []
                other_indices = []

                for j, (other_orig, other_norm) in enumerate(values_to_compare):
                    if i != j:  # исключаем текущий элемент
                        other_values.append(other_norm)
                        other_indices.append(j)

                if not other_values:
                    continue

                # Используем fuzzy matching
                if use_multiple_metrics:
                    # Используем несколько метрик и усредняем результат
                    scores = []

                    # 1. Основное сравнение (fuzz.ratio)
                    result1 = process.extractOne(norm_value, other_values, scorer=fuzz.ratio)
                    if result1:
                        match1, score1, idx1 = result1
                        scores.append(score1)

                    # 2. Частичное сравнение (fuzz.partial_ratio) - для длинных строк
                    result2 = process.extractOne(norm_value, other_values, scorer=fuzz.partial_ratio)
                    if result2:
                        match2, score2, idx2 = result2
                        scores.append(score2)

                    # 3. Сравнение с сортировкой токенов (fuzz.token_sort_ratio)
                    result3 = process.extractOne(norm_value, other_values, scorer=fuzz.token_sort_ratio)
                    if result3:
                        match3, score3, idx3 = result3
                        scores.append(score3)

                    if scores:
                        avg_score = np.mean(scores)
                        best_idx = result1[2] if result1 else result2[2] if result2 else result3[2]

                        if avg_score > fuzzy_threshold * 100:
                            # Находим оригинальное значение для исправления
                            best_orig = values_to_compare[other_indices[best_idx]][0]
                            if orig_value != best_orig:
                                col_corrections[orig_value] = best_orig
                else:
                    # Используем только одну метрику (по умолчанию fuzz.ratio)
                    result = process.extractOne(norm_value, other_values, scorer=fuzz.ratio)

                    if result:
                        match, score, idx = result

                        if score > fuzzy_threshold * 100:
                            # Находим оригинальное значение для исправления
                            best_orig = values_to_compare[other_indices[idx]][0]
                            if orig_value != best_orig:
                                col_corrections[orig_value] = best_orig

            if col_corrections:
                correction_dict[col] = col_corrections

    # Применяем исправления
    for col, corrections in correction_dict.items():
        if col not in X_train_corrected.columns:
            continue

        # Применяем к train
        X_train_corrected[col] = X_train_corrected[col].replace(corrections)

        # Применяем к test
        if col in X_test_corrected.columns:
            X_test_corrected[col] = X_test_corrected[col].replace(corrections)

        # Сохраняем информацию
        correction_info[col] = {
            'corrections': corrections,
            'count_corrected_train': (X_train[col].isin(corrections.keys())).sum(),
            'count_corrected_test': (X_test[col].isin(corrections.keys())).sum() if col in X_test.columns else 0,
            'normalize_used': normalize,
            'multiple_metrics_used': use_multiple_metrics
        }

    return X_train_corrected, X_test_corrected, correction_info


# Универсальная функция обработки (объединение 4х) с возможностью выбора
def process_categorical_outliers(X_train, X_test, strategy='group', detection_info=None, **kwargs):
    """
    Универсальная функция обработки категориальных выбросов

    Parameters:
    -----------
    X_train, X_test : DataFrame
    strategy : str
        'group' - группировка редких категорий
        'mode' - замена на моду
        'flag' - создание бинарных признаков
        'correct' - исправление опечаток
        'combined' - комбинированная стратегия
    detection_info : dict
        Результат detection_categorical_outliers
    **kwargs : дополнительные параметры для конкретных стратегий

    Returns:
    --------
    X_train_processed, X_test_processed : DataFrames
    processing_info : dict, информация об обработке
    """
    # Если detection_info не передан, вычисляем его
    if detection_info is None and strategy != 'correct':
        _, detection_info = detection_categorical_outliers(X_train)

    processing_info = {
        'strategy': strategy,
        'original_shape': (X_train.shape, X_test.shape)
    }

    # Применяем выбранную стратегию
    if strategy == 'group':
        X_train_processed, X_test_processed, info = group_rare_categories(
            X_train, X_test, detection_info,
            threshold=kwargs.get('threshold', 0.01),
            replacement=kwargs.get('replacement', 'Other')
        )
        processing_info['method_info'] = info

    elif strategy == 'mode':
        X_train_processed, X_test_processed, info = replace_with_mode(
            X_train, X_test, detection_info
        )
        processing_info['method_info'] = info

    elif strategy == 'flag':
        X_train_processed, X_test_processed, info = add_outlier_flags(
            X_train, X_test, detection_info,
            prefix=kwargs.get('prefix', 'outlier_')
        )
        processing_info['method_info'] = info

    elif strategy == 'correct':
        X_train_processed, X_test_processed, info = correct_typos(
            X_train, X_test,
            correction_dict=kwargs.get('correction_dict', None),
            fuzzy_threshold=kwargs.get('fuzzy_threshold', 0.8)
        )
        processing_info['method_info'] = info

    elif strategy == 'combined':
        # Применяем несколько стратегий последовательно
        X_temp_train = X_train.copy()
        X_temp_test = X_test.copy()
        combined_info = {}

        # 1. Исправляем опечатки
        X_temp_train, X_temp_test, correct_info = correct_typos(
            X_temp_train, X_temp_test,
            correction_dict=kwargs.get('correction_dict', None)
        )
        combined_info['correction'] = correct_info

        # 2. Обновляем detection_info после исправлений
        _, updated_detection_info = detection_categorical_outliers(X_temp_train)

        # 3. Группируем редкие категории
        X_temp_train, X_temp_test, group_info = group_rare_categories(
            X_temp_train, X_temp_test, updated_detection_info,
            threshold=kwargs.get('threshold', 0.01)
        )
        combined_info['grouping'] = group_info

        # 4. Добавляем флаги (опционально)
        if kwargs.get('add_flags', False):
            X_temp_train, X_temp_test, flag_info = add_outlier_flags(
                X_temp_train, X_temp_test, updated_detection_info
            )
            combined_info['flags'] = flag_info

        X_train_processed, X_test_processed = X_temp_train, X_temp_test
        processing_info['method_info'] = combined_info

    else:
        raise ValueError(f"Неизвестная стратегия: {strategy}. "
                         f"Используйте 'group', 'mode', 'flag', 'correct' или 'combined'")

    processing_info['processed_shape'] = (X_train_processed.shape, X_test_processed.shape)

    return X_train_processed, X_test_processed, processing_info


# Обрабатываем численные выбросы

# Определяем какие строчки в train у нас выбросы, а так же границы
mask, lower_bounds, upper_bounds = detection_MAD(X_train)
# Удаляем строки с выбросами из X_train, чтобы получить чистые данные
X_train = deleting_outliers_train(X_train, mask)
# Удаляем соответствующие строки в y_train
y_train = y_train.loc[~mask]

# Обрабатываем выбросы в X_test
# Так как работаем с бизнес логикой, наши высокие значения сборов - не выбросы, а очень лояльные клиенты, которых нам надо удерживать
X_test = capping_outliers_test(X_train, X_test, lower_bounds, upper_bounds,
                               ["tenure", "MonthlyCharges", "TotalCharges"])

# Отработка категориальных выбросов
if detection_categorical_outliers(X_train)[0].sum():
    X_train, X_test, _ = process_categorical_outliers(X_train, X_test, strategy="combined")
