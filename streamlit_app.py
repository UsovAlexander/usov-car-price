import io
import re
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import phik

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Car Price", layout="wide")

def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            for i in range(0, len(df)):
                try:
                    ffil_value = float(df.loc[i, 'max_power'][:-4])
                    df.loc[i, 'max_power'] = ffil_value

                except:
                    if df.loc[i, 'max_power'] in [np.nan, 'nan']:
                        continue
                    elif df.loc[i, 'max_power'] == '0':
                        ffil_value = float(df.loc[i, 'max_power'])
                        df.loc[i, 'max_power'] = ffil_value
                    else:
                        df.loc[i, 'max_power'] = 0

            return df
        except Exception as e:
            st.error(f"Ошибка загрузки файла: {e}")
            return None
    return None

def basic_info(df: pd.DataFrame):
    st.subheader("Базовая информация о данных")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Размерность", f"{df.shape[0]} строк × {df.shape[1]} столбцов")
    
    with col2:
        duplicates = df.duplicated().sum()
        st.metric("Дубликаты", duplicates)
    
    with col3:
        missing = df.isna().sum().sum()
        st.metric("Пропуски", missing)
    
    with col4:
        memory_usage = f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        st.metric("Память", memory_usage)
    
    st.subheader("Детальная информация")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Типы данных:**")
        dtype_info = pd.DataFrame(df.dtypes, columns=['Тип данных'])
        st.dataframe(dtype_info, width='stretch')
    
    with col2:
        st.write("**Статистика пропусков:**")
        missing_df = pd.DataFrame({
            'Столбец': df.columns,
            'Пропуски': df.isna().sum(),
            '% пропусков': (df.isna().sum() / len(df) * 100).round(2)
        })
        st.dataframe(missing_df, width='stretch')
    
    st.subheader("Описательная статистика (числовые признаки)")
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        st.dataframe(df[num_cols].describe(), width='stretch')
    
    st.subheader("Описательная статистика (категориальные признаки)")
    cat_cols = df.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        st.dataframe(pd.DataFrame(df[cat_cols].describe()), width='stretch')

def convert_cat_col_to_num(df: pd.DataFrame) -> pd.DataFrame:
    cols = ['mileage', 'engine', 'max_power']
    for col in cols:
      df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='raise').astype(float)

    # функция доработана при помощи deepseek
    def extract_simple(torque_str):
        if pd.isna(torque_str):
            return pd.Series([np.nan, np.nan])

        s = str(torque_str).lower()
        s = s.replace(',', '')
        s = re.sub(r'[()]', ' ', s)

        # Извлекаем все числа
        numbers = re.findall(r'\d+\.?\d*', s)
        numbers = [float(x) for x in numbers]

        # Извлекаем единицы измерения - ищем в исходной строке ДО очистки
        has_kgm = 'kgm' in s
        has_nm = 'nm' in s or ('n' in s and 'm' in s and 'kgm' not in s)

        torque_value = np.nan
        max_rpm = np.nan

        if numbers:
            # Первое число - обычно значение момента
            torque_value = numbers[0]
            if has_kgm:
                torque_value *= 9.807

            # Ищем RPM (обычно последнее число или из диапазона)
            if len(numbers) >= 2:
                if 'rpm' in s:
                    # Находим ВСЕ числа перед RPM
                    parts_before_rpm = s.split('rpm')[0]
                    rpm_numbers = re.findall(r'\d+\.?\d*', parts_before_rpm)
                    if rpm_numbers:
                        rpm_numbers = [float(x) for x in rpm_numbers]
                        # Ищем числа, которые могут быть оборотами (обычно > 1000)
                        potential_rpm = [x for x in rpm_numbers if x > 500]
                        if potential_rpm:
                            max_rpm = max(potential_rpm)

                # Если RPM не найдены, но есть несколько чисел
                if pd.isna(max_rpm):
                    # Ищем числа, которые могут быть оборотами (исключаем значение момента)
                    potential_rpm = [x for x in numbers[1:] if x > 500 and x <= 10000]
                    if potential_rpm:
                        max_rpm = max(potential_rpm)
                    else:
                        # Берем последнее число как запасной вариант
                        max_rpm = numbers[-1]

        return pd.Series([torque_value, max_rpm])

    df[['torque', 'max_torque_rpm']] = df['torque'].apply(extract_simple)

    return df

def preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset=df.drop('selling_price', axis=1).columns).reset_index(drop=True)
    df = convert_cat_col_to_num(df)
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df[['engine', 'seats']] = df[['engine', 'seats']].astype(int)

    return df

def plot_numerical(df: pd.DataFrame, column: str):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))
    
    data = df[column].dropna()
    
    x_min = data.min()
    x_max = data.max()
    
    ax1.hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_title(f'Гистограмма - {column}')
    ax1.set_ylabel('Частота')
    ax1.set_xlim(x_min, x_max)
    ax1.grid(True, alpha=0.3)
    
    ax2.boxplot(data, vert=False)
    ax2.set_title(f'Боксплот - {column}')
    ax2.set_xlabel(column)
    ax2.set_xlim(x_min, x_max)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)

def plot_categorical(df: pd.DataFrame, column: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    value_counts = df[column].value_counts()
    
    top_10 = value_counts.head(10)
    ax1.pie(top_10.values, labels=top_10.index, autopct='%1.1f%%', startangle=90)
    ax1.set_title(f'Круговая диаграмма (топ-10) - {column}')
    
    ax2.bar(range(len(top_10)), top_10.values, color='lightgreen')
    ax2.set_title(f'Столбчатая диаграмма (топ-10) - {column}')
    ax2.set_xlabel(column)
    ax2.set_ylabel('Количество')
    ax2.set_xticks(range(len(top_10)))
    ax2.set_xticklabels(top_10.index, rotation=45, ha='right')
    ax2.grid()
    
    plt.tight_layout()
    st.pyplot(fig)

def apply_transformation(x, transformation):
    if transformation == 'Исходные данные':
        return x
    elif transformation == 'log(x)':
        return np.log(np.abs(x) + 1e-9)
    elif transformation == 'exp(x)':
        return np.exp(x)
    elif transformation == '1/x':
        return 1 / (np.abs(x) + 1e-9)
    elif transformation == 'x²':
        return x ** 2
    elif transformation == 'x³':
        return x ** 3
    elif transformation == '√x':
        return np.sqrt(np.abs(x))
    elif transformation == 'Стандартизация':
        return (x - x.mean()) / x.std()
    elif transformation == 'Нормализация':
        return (x - x.min()) / (x.max() - x.min())
    return x

def correlation_matrix(df: pd.DataFrame, method: str='pearson'):
    st.subheader(f"Матрица корреляций ({method})")
    
    if method == 'phik':
        corr_matrix = df.drop('name', axis=1).phik_matrix()
    else:
        if method == 'pearson':
            numeric_df = df.select_dtypes(include=[np.number])
            corr_matrix = numeric_df.corr(method='pearson')
        else:
            numeric_df = df.select_dtypes(include=[np.number])
            corr_matrix = numeric_df.corr(method='spearman')
    
    fig = plt.figure(figsize=(20, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='Blues')
    plt.title(f'Матрица корреляций ({method.upper()})')
    st.pyplot(fig)
    
    st.subheader("Топ коррелированных пар")
    corr_pairs = corr_matrix.unstack().sort_values(ascending=False)
    top_pairs = pd.DataFrame(corr_pairs[corr_pairs < 0.99].head(10), columns=['Корреляция'])
    st.dataframe(top_pairs, width='stretch')

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
  df['engine * max_power'] = df['engine'] * df['max_power']
  df['year sq'] = df['year'] ** 2
  df['mod_torque'] = np.log(df['torque'])
  df['gas'] = df['fuel'].apply(lambda x: 1 if x in ['CNG', 'LPG'] else 0)
  df['model'] = df['name'].apply(lambda x: x.split()[1])
  df['name'] = df['name'].apply(lambda x: x.split()[0])

  return df

def main():
    if 'cleaned_df' not in st.session_state:
        st.session_state.cleaned_df = None

    st.title("Анализ данных автомобилей и предсказание цен")
    st.markdown("---")
    
    st.sidebar.header("Загрузка данных")
    uploaded_file = st.sidebar.file_uploader("Загрузите CSV файл", type=['csv'])
    
    if uploaded_file is None:
        st.info("Пожалуйста, загрузите CSV файл через sidebar чтобы начать анализ")
        return
    
    df = load_data(uploaded_file)
    if df is None:
        return

    tab1, tab2, tab3, tab4 = st.tabs([
        "Общая информация", 
        "Анализ признаков", 
        "Корреляции",
        "Предсказание цен"
    ])
    
    with tab1:
        basic_info(df)
        st.info("Для корректного отображения графиков рекомендуется сделать предобработку данных.")
        if st.button("Предобработка данных", key="preprocessing", type='primary'):
            st.session_state.cleaned_df = preprocessing(df.copy())
            st.success("Данные успешно предобработаны!")
            st.rerun()
    
    with tab2:
        st.header("Детальный анализ признаков")
        if st.session_state.cleaned_df is not None:
            df = st.session_state.cleaned_df
        
        selected_column = st.selectbox("Выберите признак для анализа:", df.columns)
        cat_cols = df.select_dtypes(include='object').columns.tolist() + ['seats']
        num_cols = [col for col in df.columns if col not in cat_cols]

        
        if selected_column:
            col_type = 'Числовой' if selected_column in num_cols else 'Категориальный'
            st.write(f"**Тип признака:** {col_type}")
            
            if col_type == 'Числовой':
                plot_numerical(df, selected_column)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Среднее", f"{df[selected_column].mean():.2f}")
                with col2:
                    st.metric("Медиана", f"{df[selected_column].median():.2f}")
                with col3:
                    st.metric("Стандартное отклонение", f"{df[selected_column].std():.2f}")
                
            else:
                plot_categorical(df, selected_column)
                value_counts = df[selected_column].value_counts()
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Топ-5 значений:**")
                    st.dataframe(value_counts.head(), width='stretch')
                with col2:
                    st.write("**Статистика:**")
                    st.metric("Уникальных значений", value_counts.shape[0])
                    st.metric("Самое частое", value_counts.index[0])
                    st.metric("Частота самого частого", value_counts.iloc[0])
        
        st.subheader("Распределение данных между числовыми признаками.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_axis = st.selectbox("Ось X:", df.select_dtypes(include=[np.number]).columns)
        with col2:
            y_axis = st.selectbox("Ось Y:", df.select_dtypes(include=[np.number]).columns)
        with col3:
            transformations = [
                'Исходные данные', 'log(x)', 'exp(x)', '1/x', 'x²', 'x³', 
                '√x', 'Стандартизация', 'Нормализация'
            ]
            x_transform = st.selectbox("Преобразование X:", transformations)
        
        if x_axis and y_axis:
            fig = plt.figure(figsize=(20, 10))
            
            x_data = apply_transformation(df[x_axis], x_transform)
            y_data = df[y_axis]
            
            scatter = plt.scatter(x_data, y_data)
            plt.xlabel(f"{x_axis} ({x_transform})")
            plt.ylabel(y_axis)
            plt.title(f"Scatter Plot: {x_axis} vs {y_axis}")
            plt.grid()
            st.pyplot(fig)
    
    with tab3:
        st.header("Анализ корреляций")
        
        correlation_method = st.radio(
            "Выберите метод корреляции:",
            ['pearson', 'spearman', 'phik'],
            horizontal=True
        )
        
        correlation_matrix(df, correlation_method)
    
    with tab4:
        st.header("Предсказание цен на автомобили")
        
        try:
            with open('feature_engineering_ridge_grid.pkl', 'rb') as file:
                model = pickle.load(file)
            st.success("Модель успешно загружена!")
        except FileNotFoundError:
            st.error("Файл модели 'feature_engineering_ridge_grid.pkl' не найден!")
        
        pred_tab1, pred_tab2 = st.tabs(["Ручной ввод", "Загрузка файла"])
        
        with pred_tab1:
            st.subheader("Введите параметры автомобиля:")

            col1, col2 = st.columns(2)
            
            with col1:
                name = st.selectbox("Марка", df['name'].unique() if 'name' in df.columns else [])
                year = st.number_input("Год выпуска", min_value=1990, max_value=2024, value=2018, help="Пример: 2018")
                km_driven = st.number_input("Пробег (км)", min_value=0, value=50000, help="Пример: 50000")
                fuel = st.selectbox("Топливо", df['fuel'].unique() if 'fuel' in df.columns else [])
                seller_type = st.selectbox("Тип продавца", df['seller_type'].unique() if 'seller_type' in df.columns else [])
                transmission = st.selectbox("Трансмиссия", df['transmission'].unique() if 'transmission' in df.columns else [])
                owner = st.selectbox("Владелец", df['owner'].unique() if 'owner' in df.columns else [])
            with col2:
                mileage = st.number_input("Расход топлива", min_value=0.0, value=20.0, step=0.1, help="Пример: 20.0 км/л")
                engine = st.number_input("Объем двигателя (см³)", min_value=0, value=1200, help="Пример: 1200")
                max_power = st.number_input("Мощность (л.с.)", min_value=0.0, value=80.0, step=0.1, help="Пример: 80.0")
                torque = st.number_input("Крутящий момент (Нм)", min_value=0.0, value=150.0, step=0.1, help="Пример: 150.0")
                seats = st.number_input("Количество мест", min_value=2, max_value=10, value=5)
                max_torque_rpm = st.number_input("Макс. крутящий момент (об/мин)", min_value=0, value=3000, help="Пример: 3000")
            
            if st.button("Предсказать цену", type="primary"):
                input_data = pd.DataFrame({
                    'name': [name],
                    'year': [year],
                    'km_driven': [km_driven],
                    'fuel': [fuel],
                    'seller_type': [seller_type],
                    'transmission': [transmission],
                    'owner': [owner],
                    'mileage': [mileage],
                    'engine': [engine],
                    'max_power': [max_power],
                    'torque': [torque],
                    'seats': [seats],
                    'max_torque_rpm': [max_torque_rpm]
                })
                
                try:
                    prediction = model.predict(input_data)[0]
                    st.success(f"Предсказанная цена: **{prediction:,.2f}**")
                    
                    st.info(f"""
                    **Введенные параметры:**
                    - Марка: {name}
                    - Год: {year}
                    - Пробег: {km_driven:,} км
                    - Мощность: {max_power} л.с.
                    - Объем двигателя: {engine} см³
                    """)
                    
                except Exception as e:
                    st.error(f"Ошибка при предсказании: {e}")
        
        with pred_tab2:
            st.subheader("Загрузите файл для пакетного предсказания")
            
            prediction_file = st.file_uploader("Загрузите CSV файл с данными для предсказания", 
                                            type=['csv'], key="prediction")
            
            if prediction_file is not None:
                pred_df = load_data(prediction_file)
                
                if pred_df is not None:
                    if 'selling_price' in pred_df.columns:
                        pred_df = pred_df.drop('selling_price', axis=1)
                    pred_df = convert_cat_col_to_num(pred_df)
                    # пришлось заколхозить, т.к. seats в пайплайн идет как категориальный признак и пропуски будут заполняться заглушкой, а не медианой
                    pred_df['seats'] = pred_df['seats'].fillna(df['seats'].median())
                    pred_df['seats'] = pred_df['seats'].astype(int)
                    
                    st.write("**Предпросмотр данных:**")
                    st.dataframe(pred_df.head(), width='stretch')
                    
                    if st.button("Выполнить пакетное предсказание", type="primary"):
                        try:
                            predictions = model.predict(pred_df)
                            result_df = pred_df.copy()
                            result_df['predicted_price'] = predictions
                            
                            st.success(f"Предсказания выполнены для {len(predictions)} записей")
                            
                            st.write("**Результаты предсказаний:**")
                            st.dataframe(result_df, width='stretch')
                            
                            csv_file = result_df.to_csv(index=False)
                            st.download_button(
                                label="Скачать результаты в CSV",
                                data=csv_file,
                                file_name="car_price_predictions.csv",
                                mime="text/csv"
                            )
                            
                        except Exception as e:
                            st.error(f"Ошибка при пакетном предсказании: {e}")

if __name__ == "__main__":
    main()