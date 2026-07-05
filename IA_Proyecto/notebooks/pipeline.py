import logging
import os
import re
import shutil
import pandas as pd
from sklearn.preprocessing import StandardScaler

# --- Configuración Dinámica de Rutas (Solución al problema de ejecución) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
logging.basicConfig(
    filename=os.path.join(BASE_DIR, "logs/ingesta.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

origen = os.path.join(BASE_DIR, "../../Telco.csv")
destino = os.path.join(BASE_DIR, "../data/raw/Telco.csv")

try:
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    shutil.copy(origen, destino)
    logging.info("Archivo copiado correctamente")
    print("Ingesta completada")
except Exception as e:
    logging.error(f"Error en ingesta: {e}")
    print(f"Error: {e}")

df = pd.read_csv(destino)
df2 = pd.read_csv(destino)


# --- Funciones de Optimización ---


def clean_and_impute_data(df_raw, model_type="logistic"):
    df_clean = df_raw.copy()

    text_cols = df_clean.select_dtypes(include=["object", "string"]).columns
    for col in text_cols:
        df_clean[col] = df_clean[col].astype(str).str.strip()

    df_clean["TotalCharges"] = pd.to_numeric(
        df_clean["TotalCharges"], errors="coerce"
    )

    df_clean.loc[df_clean["tenure"] == 0, "TotalCharges"] = df_clean.loc[
        df_clean["tenure"] == 0, "TotalCharges"
    ].fillna(0)

    mask_missing_total = (df_clean["TotalCharges"].isna()) & (
        df_clean["tenure"] > 0
    )
    df_clean.loc[mask_missing_total, "TotalCharges"] = (
        df_clean.loc[mask_missing_total, "tenure"]
        * df_clean.loc[mask_missing_total, "MonthlyCharges"]
    )

    mask_negative_monthly = df_clean["MonthlyCharges"] <= 0
    df_clean.loc[mask_negative_monthly, "MonthlyCharges"] = df_clean.apply(
        lambda row: (
            row["TotalCharges"] / row["tenure"]
            if row["tenure"] > 0 and row["MonthlyCharges"] <= 0
            else row["MonthlyCharges"]
        ),
        axis=1,
    )
    
    median_monthly = df_clean["MonthlyCharges"].median()
    df_clean["MonthlyCharges"] = (
        df_clean["MonthlyCharges"]
        .fillna(median_monthly)
        .mask(df_clean["MonthlyCharges"] <= 0, median_monthly)
    )

    categorical_cols = df_clean.select_dtypes(include=["object", "string"]).columns
    for col in categorical_cols:
        df_clean[col] = df_clean[col].replace(["", "nan", "None"], pd.NA)

        if df_clean[col].isna().sum() > 0:
            if model_type == "rf":
                df_clean[col] = df_clean[col].fillna("Unknown")
            else:
                mode_val = df_clean[col].mode()[0]
                df_clean[col] = df_clean[col].fillna(mode_val)

    return df_clean


def to_snake_case(text):
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    s = s.replace(" ", "_")
    s = s.lower()
    return re.sub(r"_+", "_", s)


def rename_datasets_automatically(dfs):
    for df_item in dfs:
        df_item.columns = [to_snake_case(col) for col in df_item.columns]
    return dfs


def detect_low_cardinality_cols(df, max_unique=3):
    selected_cols = []
    for col in df.columns:
        # Detectar tanto 'object' como el nuevo tipo 'string' nativo de Pandas
        if df[col].dtype == "object" or pd.api.types.is_string_dtype(df[col]):
            unique_values = set()
            for value in df[col]:
                if pd.notna(value):
                    unique_values.add(value)
                if len(unique_values) > max_unique:
                    break
            if len(unique_values) <= max_unique:
                selected_cols.append(col)
    return selected_cols


def drop_and_clean(df, cols_to_drop, clean_total=False):
    df_clean = df.drop(columns=cols_to_drop, errors="ignore").copy()
    if clean_total and "total_charges" in df_clean.columns:
        df_clean["total_charges"] = (
            pd.to_numeric(df_clean["total_charges"], errors="coerce").fillna(0)
        )
    return df_clean


def encode_categorical(dfs, mapping_schema):
    for df_item in dfs:
        for col, mapper in mapping_schema.items():
            if col in df_item.columns:
                df_item[col] = df_item[col].map(mapper).astype(int)
    return dfs


def validate_dataframe(
    df, expected_column_count, check_financial=False, name="Dataset", scaled=False
):
    errors = []
    warnings = []
    cols_present = set(df.columns)

    print(f"\n=== Iniciando Validación de {name} ===")

    if len(df.columns) != expected_column_count:
        errors.append(
            f"Se esperaban {expected_column_count} columnas, pero se detectaron {len(df.columns)}."
        )

    total_nulls = df.isnull().sum().sum()
    if total_nulls > 0:
        errors.append(f"Se encontraron {total_nulls} nulos post-procesamiento.")

    service_cols = {
        "internet_service", "online_security", "online_backup", 
        "device_protection", "tech_support", "streaming_tv", "streaming_movies"
    }
    if service_cols.issubset(cols_present):
        sin_internet = df[df["internet_service"] == 0]
        cols_internet = [
            "online_security", "online_backup", "device_protection", 
            "tech_support", "streaming_tv", "streaming_movies"
        ]
        for col in cols_internet:
            inconsistentes = sin_internet[sin_internet[col] != 2]
            if len(inconsistentes) > 0:
                warnings.append(
                    f"{col}: {len(inconsistentes)} clientes sin internet con valor != 2"
                )

    if {"phone_service", "multiple_lines"}.issubset(cols_present):
        sin_phone = df[df["phone_service"] == 0]
        inconsistentes = sin_phone[sin_phone["multiple_lines"] != 2]
        if len(inconsistentes) > 0:
            warnings.append(
                f"multiple_lines: {len(inconsistentes)} clientes sin phone_service con valor != 2"
            )

    # Omitir la regla '<= 0' si la columna ya pasó por el StandardScaler
    if not scaled and "monthly_charges" in df.columns:
        fuera = df[df["monthly_charges"] <= 0]
        if len(fuera) > 0:
            errors.append(f"monthly_charges: {len(fuera)} valores <= 0")

    if check_financial and "total_charges" in df.columns:
        fuera = df[df["total_charges"].notna() & (df["total_charges"] < 0)]
        if len(fuera) > 0:
            errors.append(f"total_charges: {len(fuera)} valores negativos")

        if "tenure" in df.columns and "monthly_charges" in df.columns:
            condicion_total_menor = (df["tenure"] >= 1) & (
                df["total_charges"] < df["monthly_charges"]
            )
            df_error_financiero = df[condicion_total_menor]
            if len(df_error_financiero) > 0:
                errors.append(
                    f"Inconsistencia financiera cruzada: {len(df_error_financiero)} registros con total_charges < monthly_charges."
                )

    CATEGORICAS = {
        "gender": {0, 1}, "senior_citizen": {0, 1}, "partner": {0, 1}, "dependents": {0, 1},
        "phone_service": {0, 1}, "paperless_billing": {0, 1}, "churn": {0, 1},
        "multiple_lines": {0, 1, 2}, "online_security": {0, 1, 2}, "online_backup": {0, 1, 2},
        "device_protection": {0, 1, 2}, "tech_support": {0, 1, 2}, "streaming_tv": {0, 1, 2},
        "streaming_movies": {0, 1, 2}, "internet_service": {0, 1, 2}, "contract": {0, 1, 2},
        "payment_method": {0, 1, 2, 3},
    }
    for col, valores_validos in CATEGORICAS.items():
        if col in df.columns:
            valores_reales = set(df[col].dropna().unique())
            invalidos = valores_reales - valores_validos
            if invalidos:
                errors.append(
                    f"{col}: valores numéricos inválidos detectados: {invalidos}"
                )

    if warnings:
        for w in warnings:
            logging.warning(f"[{name}] {w}")
            print(f"WARNING: [{name}] {w}")

    if errors:
        for e in errors:
            logging.error(f"[{name}] {e}")
            print(f"ERROR: [{name}] {e}")
        raise ValueError(
            f"Fallo crítico en control de calidad para: {name}. Proceso abortado."
        )

    print(f"OK - {name} validado correctamente.")
    logging.info(f"Validación exitosa para {name}")


# --- Ejecución del Pipeline Seguro ---

try:
    logging.info("Iniciando fase de transformación de datos...")

    df = clean_and_impute_data(df, model_type="logistic")
    df2 = clean_and_impute_data(df2, model_type="rf")
    logging.info("Limpieza e imputación cruzada completada")

    df, df2 = rename_datasets_automatically([df, df2])
    logging.info("1: Titulos cambiados a snake_case")

    df = drop_and_clean(df, cols_to_drop=["customer_id", "total_charges"])
    df2 = drop_and_clean(df2, cols_to_drop=["customer_id"], clean_total=True)
    logging.info("2: Columna customer_id eliminada para anonimización")

    scaler = StandardScaler()
    df[["tenure", "monthly_charges"]] = scaler.fit_transform(
        df[["tenure", "monthly_charges"]]
    )
    logging.info("3: Escalado numérico (StandardScaler) aplicado a variables continuas")

    dynamic_binary_cols = detect_low_cardinality_cols(df, max_unique=3)
    logging.info(f"4: Columnas de baja cardinalidad detectadas: {dynamic_binary_cols}")

    standard_map = {"Yes": 1, "No": 0, "No internet service": 2}
    master_mapping = {col: standard_map for col in dynamic_binary_cols}
    master_mapping.update(
        {
            "gender": {"Male": 1, "Female": 0},
            "multiple_lines": {"No": 0, "Yes": 1, "No phone service": 2},
            "internet_service": {"No": 0, "DSL": 1, "Fiber optic": 2},
            "contract": {"Month-to-month": 0, "One year": 1, "Two year": 2},
            "payment_method": {
                "Electronic check": 0,
                "Mailed check": 1,
                "Bank transfer (automatic)": 2,
                "Credit card (automatic)": 3,
            },
        }
    )

    df, df2 = encode_categorical([df, df2], master_mapping)
    logging.info("6: Mapeo y codificación masiva completada")

    # 7. Gate de Validación (Se añade el parámetro 'scaled' para evitar falsos negativos)
    validate_dataframe(
        df,
        expected_column_count=19,
        check_financial=False,
        name="DF_Logistic_Regression",
        scaled=True,
    )
    validate_dataframe(
        df2,
        expected_column_count=20,
        check_financial=True,
        name="DF_Random_Forest",
        scaled=False,
    )

    # 8. Carga / Almacenamiento Seguro
    output_dir = os.path.join(BASE_DIR, "../data/processed")
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "Telco_clean.csv"), index=False)
    df2.to_csv(os.path.join(output_dir, "Telco_limpio.csv"), index=False)
    logging.info("8: Datasets guardados con éxito en la ruta especificada")

    print("\nProcesamiento y validación finalizados con completo éxito.")

except ValueError as val_err:
    logging.critical(f"Pipeline interrumpido por validación: {val_err}")
    print(f"\nCRÍTICO: Los archivos NO se guardaron. Razón: {val_err}")

except Exception as e:
    logging.critical(f"Fallo inesperado del sistema: {e}")
    print(f"\nERROR INESPERADO: {e}")