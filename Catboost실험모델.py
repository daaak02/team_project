import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
import holidays
import warnings
warnings.filterwarnings("ignore")

# ===========================
# CSV 로드
# ===========================
def load_csv_with_fallback(path):
    encodings = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"✓ 로드 완료: {path} ({enc})")
            return df
        except:
            pass
    print(f"※ fallback(latin1) 로드: {path}")
    return pd.read_csv(path, encoding="latin1")


# ===========================
# 전처리 클래스
# ===========================
class PowerPreprocessorCat:

    def __init__(self):
        self.kr_holidays = holidays.KR()

    def preprocess_datetime(self, df):
        df = df.copy()
        df["일시"] = pd.to_datetime(df["일시"])
        df["year"] = df["일시"].dt.year
        df["month"] = df["일시"].dt.month
        df["day"] = df["일시"].dt.day
        df["hour"] = df["일시"].dt.hour
        df["dayofweek"] = df["일시"].dt.dayofweek
        df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
        df["is_holiday"] = df["일시"].dt.date.apply(lambda x: int(x in self.kr_holidays))
        return df

    def merge_building_info(self, df, building_info):
        return df.merge(building_info, on="건물번호", how="left")

    def handle_missing_and_rename(self, df):
        df = df.copy()

        df = df.rename(columns={
            "기온(°C)": "기온",
            "습도(%)": "습도",
            "풍속(m/s)": "풍속",
            "강수량(mm)": "강수량",
            "일조(hr)": "일조",
            "일사(MJ/m2)": "일사"
        })

        df = df.replace("-", np.nan)

        weather_cols = ["기온", "강수량", "풍속", "습도", "일조", "일사"]
        for col in weather_cols:
            if col not in df.columns:
                df[col] = np.nan
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df.groupby("건물번호")[col].ffill().bfill()

        building_cols = ["연면적(m2)", "냉방면적(m2)",
                         "태양광용량(kW)", "ESS저장용량(kWh)", "PCS용량(kW)"]
        for col in building_cols:
            if col not in df.columns:
                df[col] = np.nan
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col].fillna(df[col].median(), inplace=True)

        return df

    def create_features(self, df):
        df = df.copy()

        if "기온" in df.columns and "습도" in df.columns:
            df["THI"] = (
                9/5 * df["기온"]
                - 0.55 * (1 - df["습도"]/100) * (9/5 * df["기온"] - 26)
                + 32
            )

        if "기온" in df.columns and "풍속" in df.columns:
            df["wind_chill"] = 13.12 + 0.6215 * df["기온"] - 11.37 * (df["풍속"] ** 0.16)

        if "연면적(m2)" in df.columns and "냉방면적(m2)" in df.columns:
            df["cooling_ratio"] = df["냉방면적(m2)"] / (df["연면적(m2)"] + 1)

        if "연면적(m2)" in df.columns and "태양광용량(kW)" in df.columns:
            df["solar_density"] = df["태양광용량(kW)"] / (df["연면적(m2)"] + 1)

        if "기온" in df.columns:
            df["temp_sq"] = df["기온"] ** 2

        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        df["solar_effect"] = df["일사"] * df["태양광용량(kW)"]

        return df

    def add_lag_features(self, df):
        df = df.copy()
        for lag in [1, 24, 168]:
            df[f"lag_{lag}"] = df.groupby("건물번호")["전력소비량"].shift(lag)
        return df

    def full_pipeline(self, train, building_info):
        train = self.preprocess_datetime(train)
        train = self.merge_building_info(train, building_info)
        train = self.handle_missing_and_rename(train)
        train = self.create_features(train)
        train = self.add_lag_features(train)
        return train


# ===========================
# 이상치 제거
# ===========================
def remove_outliers_iqr(df, col, factor=1.5):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    return df[(df[col] >= lower) & (df[col] <= upper)].reset_index(drop=True)


# ===========================
# ✅ 건물별 RMSE
# ===========================
def evaluate_catboost_by_building(train_raw, building_info, bld_id):

    train_sub = train_raw[train_raw["건물번호"] == bld_id].copy()
    if len(train_sub) < 500:
        return None

    train_sub = remove_outliers_iqr(train_sub, "전력소비량")

    proc = PowerPreprocessorCat()
    df = proc.full_pipeline(train_sub, building_info)
    df = df.dropna(subset=["lag_168"])

    drop_cols = [c for c in ["일시", "num_date_time"] if c in df.columns]

    X = df.drop(columns=["전력소비량"] + drop_cols)
    y = np.log1p(df["전력소비량"])

    cat_features = []
    if "건물유형" in X.columns:
        cat_features.append(X.columns.get_loc("건물유형"))

    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = CatBoostRegressor(
        iterations=1200,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=6,
        loss_function="RMSE",
        random_seed=42,
        verbose=0
    )

    model.fit(
        X_tr, y_tr,
        cat_features=cat_features,
        eval_set=(X_va, y_va),
        use_best_model=True
    )

    pred = np.expm1(model.predict(X_va))
    rmse = np.sqrt(mean_squared_error(np.expm1(y_va), pred))

    return rmse


# ===========================
# ✅ 글로벌 RMSE
# ===========================
def evaluate_catboost_global(train_raw, building_info):

    train_sub = remove_outliers_iqr(train_raw, "전력소비량")

    proc = PowerPreprocessorCat()
    df = proc.full_pipeline(train_sub, building_info)
    df = df.dropna(subset=["lag_168"])

    drop_cols = [c for c in ["일시", "num_date_time"] if c in df.columns]

    X = df.drop(columns=["전력소비량"] + drop_cols)
    y = np.log1p(df["전력소비량"])

    cat_features = []
    if "건물유형" in X.columns:
        cat_features.append(X.columns.get_loc("건물유형"))

    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=7,
        loss_function="RMSE",
        random_seed=42,
        verbose=200
    )

    model.fit(
        X_tr, y_tr,
        cat_features=cat_features,
        eval_set=(X_va, y_va),
        use_best_model=True
    )

    pred = np.expm1(model.predict(X_va))
    rmse = np.sqrt(mean_squared_error(np.expm1(y_va), pred))

    return rmse


# ===========================
# ✅ MAIN: 건물별 + 글로벌 RMSE 연속 출력
# ===========================
if __name__ == "__main__":

    train_raw = load_csv_with_fallback("C:/data/train.csv")
    building_info = load_csv_with_fallback("C:/data/building_info.csv")

    if "전력소비량(kWh)" in train_raw.columns:
        train_raw = train_raw.rename(columns={"전력소비량(kWh)": "전력소비량"})

    results = {}

    print("\n" + "="*60)
    print("▶ 건물별 시계열 CatBoost RMSE")
    print("="*60)

    for bld_id in sorted(train_raw["건물번호"].unique()):
        rmse = evaluate_catboost_by_building(train_raw, building_info, bld_id)
        if rmse is not None:
            results[bld_id] = rmse
            print(f"건물 {bld_id:>3}: RMSE = {rmse:.4f}")

    print("\n" + "="*60)
    print("▶ Global 시계열 CatBoost RMSE")
    print("="*60)

    global_rmse = evaluate_catboost_global(train_raw, building_info)
    print(f"✅ Global RMSE = {global_rmse:.4f}")

    print("\n" + "="*60)
    print("▶ 완료")
    print("="*60)
