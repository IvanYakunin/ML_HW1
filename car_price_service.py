from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib
import re
import os
from io import StringIO

# Инициализация приложения
app = FastAPI()

# Класс базового объекта
class Item(BaseModel):
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float
    manufacturer: str
    model_grouped: str

# Класс для коллекции объектов
class Items(BaseModel):
    objects: List[Item]

# Загрузка модели и метаинформации
model_path = "final_model.pkl"
columns_path = "expected_columns.pkl"
medians_path = "medians.pkl"
scaler_path = "scaler.pkl"

final_model = joblib.load(model_path)
expected_columns = joblib.load(columns_path)
medians = joblib.load(medians_path)
loaded_scaler = joblib.load(scaler_path)

# Вспомогательные функции
def preprocess_units(df):
    df['mileage'] = df['mileage'].astype(str).str.replace(r' kmpl| km/kg', '', regex=True)
    df['engine'] = df['engine'].astype(str).str.replace(r' CC', '', regex=True)
    df['max_power'] = df['max_power'].astype(str).str.replace(r' bhp', '', regex=True)
    df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce').astype(float)
    df['engine'] = pd.to_numeric(df['engine'], errors='coerce').astype(float)
    df['max_power'] = pd.to_numeric(df['max_power'], errors='coerce').astype(float)

def preprocess_torque(df):
    torque_values = []
    rpm_values = []
    for torque in df['torque']:
        torque_str = str(torque)
        torque_match = re.search(r'([\d.]+)\s*(Nm|kgm)', torque_str, re.IGNORECASE)
        torque_value = None
        if torque_match:
            torque_value = float(torque_match.group(1))
            if torque_match.group(2).lower() == 'kgm':
                torque_value *= 9.8
        rpm_match = re.search(r'@?\s*([\d,-]+)\s*rpm', torque_str, re.IGNORECASE)
        rpm_value = None
        if rpm_match:
            rpm_range = rpm_match.group(1).replace(',', '')
            if '-' in rpm_range:
                rpm_values_split = [int(r) for r in rpm_range.split('-')]
                rpm_value = sum(rpm_values_split) / len(rpm_values_split)
            else:
                rpm_value = int(rpm_range)
        torque_values.append(torque_value)
        rpm_values.append(rpm_value)
    df['torque'] = pd.to_numeric(torque_values, errors='coerce').astype(float)
    df['max_torque_rpm'] = pd.to_numeric(rpm_values, errors='coerce').astype(float)

def fill_missing_values(df, medians):
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    for column in numeric_columns:
        if column in medians.index:
            df[column] = df[column].fillna(medians[column])
    return df

def scale_numeric_features(df, scaler):
    """
    Применяет StandardScaler() к числовым столбцам в датафрейме.
    
    Параметры:
        df (pd.DataFrame): Входной датафрейм.
        
    Возвращает:
        pd.DataFrame: Датафрейм с масштабированными числовыми столбцами.
    """
    # Создаем копию входного датафрейма
    df_scaled = df.copy()
    
    # Определяем числовые столбцы
    columns = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'torque', 'max_torque_rpm']
    columns = [col for col in columns if col != "selling_price"]
    
    df_scaled[columns] = scaler.transform(df_scaled[columns])
    
    return df_scaled

def prepare_data(data: List[Item]) -> pd.DataFrame:
    df = pd.DataFrame([item.dict() for item in data])
    preprocess_units(df)
    preprocess_torque(df)
    df = fill_missing_values(df, medians)
    df = scale_numeric_features(df, loaded_scaler)
    df = pd.get_dummies(df, columns=[
        "fuel", "seller_type", "transmission", "owner", 
        "manufacturer", "model_grouped"
    ], drop_first=False)
    df = df.reindex(columns=expected_columns, fill_value=0)
    return df

# Роуты
@app.post("/predict_item")
def predict_item(item: Item):
    df = prepare_data([item])
    prediction = final_model.predict(df)
    # Преобразование предсказания в float
    return {"predicted_price": float(prediction[0])}

@app.post("/predict_items")
async def predict_items(file: UploadFile = File(...)):
    content = await file.read()
    # Чтение JSON вместо CSV
    df = pd.read_json(StringIO(content.decode('utf-8')))
    
    items = df.to_dict(orient='records')
    processed_df = prepare_data([Item(**item) for item in items])
    predictions = final_model.predict(processed_df)
    df['predicted_price'] = [float(p) for p in predictions]  # Преобразование предсказаний
    output = StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return {"file": output.getvalue()}
