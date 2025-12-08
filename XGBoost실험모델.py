import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
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
# 전처리 + 시계열
# ===========================
class PowerPreprocessorXGB:

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
        df["is_holiday"] = df["일시"].dt.date.apply(
            lambda x: int(x in self.kr_holidays)
        )
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
            df["wind_chill"] = (
                13.12 + 0.6215 * df["기온"] - 11.37 * (df["풍속"] ** 0.16)
            )

        if "연면적(m2)" in df.columns and "냉방면적(m2)" in df.columns:
            df["cooling_ratio"] = df["냉방면적(m2)"] / (df["연면적(m2)"] + 1)

        if "연면적(m2)" in df.columns and "태양광용량(kW)" in df.columns:
            df["solar_density"] = df["태양광용량(kW)"] / (df["연면적(m2)"] + 1)

        if "기온" in df.columns:
            df["temp_sq"] = df["기온"] ** 2

        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        if "일사" in df.columns and "태양광용량(kW)" in df.columns:
            df["solar_effect"] = df["일사"] * df["태양광용량(kW)"]

        return df

    def add_lag_features(self, df):
        df = df.copy()
        for lag in [1, 24, 168]:
            df[f"lag_{lag}"] = df.groupby("건물번호")["전력소비량"].shift(lag)
        return df

    def full_pipeline(self, df, building_info):
        df = self.preprocess_datetime(df)
        df = self.merge_building_info(df, building_info)
        df = self.handle_missing_and_rename(df)
        df = self.create_features(df)
        df = self.add_lag_features(df)
        return df


# ===========================
# ✅ 건물별 XGBoost + 재귀 lag 보정
# ===========================
def evaluate_xgb_with_recursive_lag(train_raw, test_raw, building_info, bld_id):

    train_sub = train_raw[train_raw["건물번호"] == bld_id].copy()
    test_sub = test_raw[test_raw["건물번호"] == bld_id].copy()

    if len(train_sub) < 500:
        return None

    proc = PowerPreprocessorXGB()
    train_feat = proc.full_pipeline(train_sub, building_info)
    train_feat = train_feat.dropna(subset=["lag_168"])

    drop_cols = [c for c in ["일시", "num_date_time"] if c in train_feat.columns]

    X_train = train_feat.drop(columns=["전력소비량"] + drop_cols)
    X_train = X_train.select_dtypes(include=[np.number])
    y_train = np.log1p(train_feat["전력소비량"])

    # ✅ 컬럼 순서 고정
    feature_cols = X_train.columns.tolist()

    model = XGBRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # ===== ✅ test 재귀 lag 보정 =====
    hist = train_sub.sort_values("일시").copy()
    preds = []

    for i in range(len(test_sub)):
        cur = test_sub.iloc[[i]].copy()
        cur = proc.preprocess_datetime(cur)
        cur = proc.merge_building_info(cur, building_info)
        cur = proc.handle_missing_and_rename(cur)
        cur = proc.create_features(cur)

        cur["lag_1"] = hist["전력소비량"].iloc[-1]
        cur["lag_24"] = hist["전력소비량"].iloc[-24]
        cur["lag_168"] = hist["전력소비량"].iloc[-168]

        X_test = cur.drop(
            columns=["전력소비량", "일시", "num_date_time"],
            errors="ignore"
        )
        X_test = X_test.select_dtypes(include=[np.number])

        # ✅ 누락 컬럼 0으로 채우고 순서 고정
        for c in feature_cols:
            if c not in X_test.columns:
                X_test[c] = 0
        X_test = X_test[feature_cols]

        pred = np.expm1(model.predict(X_test)[0])
        preds.append(pred)

        cur["전력소비량"] = pred
        hist = pd.concat([hist, cur], ignore_index=True)

    return np.mean(preds)


# ===========================
# ✅ MAIN
# ===========================
if __name__ == "__main__":

    train_raw = load_csv_with_fallback("C:/data/train.csv")
    test_raw = load_csv_with_fallback("C:/data/test.csv")
    building_info = load_csv_with_fallback("C:/data/building_info.csv")

    if "전력소비량(kWh)" in train_raw.columns:
        train_raw = train_raw.rename(
            columns={"전력소비량(kWh)": "전력소비량"}
        )

    print("\n" + "="*60)
    print("▶ 건물별 XGBoost + 시계열 lag 재귀 예측 실험")
    print("="*60)

    results = {}

    for bld_id in sorted(train_raw["건물번호"].unique()):
        pred_mean = evaluate_xgb_with_recursive_lag(
            train_raw, test_raw, building_info, bld_id
        )
        if pred_mean is not None:
            results[bld_id] = pred_mean
            print(f"건물 {bld_id:>3}: 평균 예측값 = {pred_mean:.2f}")

    print("\n" + "="*60)
    print("▶ 완료")
    print("="*60)
