import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import holidays
import warnings

warnings.filterwarnings("ignore")

# =========================================================
# 1. CSV 로드
# =========================================================
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


# =========================================================
# 2. XGBoost 전처리
# =========================================================
class PowerPreprocessorXGB:

    def __init__(self):
        self.kr_holidays = holidays.KR()
        self.encoders = {}

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

        # ✅ 일조 관련 완전 제거
        df = df.drop(columns=["일조", "일조(hr)"], errors="ignore")

        df = df.rename(columns={
            "기온(°C)": "기온",
            "습도(%)": "습도",
            "풍속(m/s)": "풍속",
            "강수량(mm)": "강수량",
            "일사(MJ/m2)": "일사"
        })

        df = df.replace("-", np.nan)

        weather_cols = ["기온", "강수량", "풍속", "습도", "일사"]
        for col in weather_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df.groupby("건물번호")[col].ffill().bfill()

        building_cols = ["연면적(m2)", "냉방면적(m2)",
                         "태양광용량(kW)", "ESS저장용량(kWh)", "PCS용량(kW)"]
        for col in building_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col].fillna(df[col].median(), inplace=True)

        return df

    def create_features(self, df):
        df = df.copy()

        if "기온" in df.columns and "습도" in df.columns:
            df["THI"] = (
                9/5*df["기온"]
                - 0.55*(1-df["습도"]/100)*(9/5*df["기온"]-26)
                + 32
            )

        if "기온" in df.columns and "풍속" in df.columns:
            df["wind_chill"] = 13.12 + 0.6215*df["기온"] - 11.37*(df["풍속"]**0.16)

        if "연면적(m2)" in df.columns and "냉방면적(m2)" in df.columns:
            df["cooling_ratio"] = df["냉방면적(m2)"] / (df["연면적(m2)"] + 1)

        if "연면적(m2)" in df.columns and "태양광용량(kW)" in df.columns:
            df["solar_density"] = df["태양광용량(kW)"] / (df["연면적(m2)"] + 1)

        if "기온" in df.columns:
            df["temp_sq"] = df["기온"] ** 2

        if "hour" in df.columns:
            df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
            df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        if "일사" in df.columns and "태양광용량(kW)" in df.columns:
            df["solar_effect"] = df["일사"] * df["태양광용량(kW)"]

        return df

    def encode_categoricals(self, train_df, test_df):
        cat_cols = train_df.select_dtypes(include=["object"]).columns

        for col in cat_cols:
            le = LabelEncoder()
            train_df[col] = train_df[col].astype(str)
            test_df[col] = test_df[col].astype(str)

            le.fit(train_df[col])
            train_df[col] = le.transform(train_df[col])
            test_df[col] = test_df[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

        return train_df, test_df

    def full_pipeline(self, train, test, building_info):
        train = self.preprocess_datetime(train)
        test = self.preprocess_datetime(test)

        train = self.merge_building_info(train, building_info)
        test = self.merge_building_info(test, building_info)

        train = self.handle_missing_and_rename(train)
        test = self.handle_missing_and_rename(test)

        train = self.create_features(train)
        test = self.create_features(test)

        train, test = self.encode_categoricals(train, test)

        return train, test


# =========================================================
# 3. 이상치 처리
# =========================================================
def remove_outliers_iqr(df, col, factor=1.5):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    return df[~((df[col] < lower) | (df[col] > upper))].reset_index(drop=True)

def remove_outliers_zscore(df, col, threshold=3.0):
    mean = df[col].mean()
    std = df[col].std()
    if std == 0:
        return df
    z = (df[col] - mean) / std
    return df[np.abs(z) <= threshold].reset_index(drop=True)

def clip_target(df, col="전력소비량", lower_q=0.01, upper_q=0.99):
    low = df[col].quantile(lower_q)
    high = df[col].quantile(upper_q)
    df[col] = df[col].clip(low, high)
    return df


# =========================================================
# 4. 사용그룹
# =========================================================
def map_usage_group(btype):
    if btype in ["호텔", "병원", "백화점", "통신시설", "전화국"]:
        return "24시간"
    elif btype in ["학교", "연구소"]:
        return "주간"
    else:
        return "기타"


# =========================================================
# 5. 그룹별 학습 + 예측
# =========================================================
def train_and_predict_for_group(train_raw_sub, test_raw_sub, building_info, version):

    train_sub = train_raw_sub.copy()
    test_sub = test_raw_sub.copy()

    if version == "B":
        train_sub = remove_outliers_iqr(train_sub, "전력소비량")
        train_sub = remove_outliers_zscore(train_sub, "전력소비량")
    elif version == "D":
        train_sub = remove_outliers_iqr(train_sub, "전력소비량")
        train_sub = remove_outliers_zscore(train_sub, "전력소비량")
        train_sub = clip_target(train_sub)

    processor = PowerPreprocessorXGB()
    train_proc, test_proc = processor.full_pipeline(train_sub, test_sub, building_info)

    drop_cols = ["일시", "num_date_time"]
    X_train = train_proc.drop(columns=drop_cols + ["전력소비량"], errors="ignore")
    y_train = train_proc["전력소비량"]
    X_test = test_proc.drop(columns=drop_cols, errors="ignore")

    X_test = X_test[X_train.columns]

    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    return pd.DataFrame({
        "num_date_time": test_sub["num_date_time"].values,
        "answer": preds
    })


# =========================================================
# 6. MAIN: D / B / D 제출 생성
# =========================================================
if __name__ == "__main__":

    train_path = "C:/data/train.csv"
    test_path = "C:/data/test.csv"
    building_path = "C:/data/building_info.csv"
    sample_path = "C:/data/sample_submission.csv"
    output_path = "C:/data/submission_xgboost_DBD_FINAL.csv"

    train_raw = load_csv_with_fallback(train_path)
    test_raw = load_csv_with_fallback(test_path)
    building_info = load_csv_with_fallback(building_path)
    sample_sub = load_csv_with_fallback(sample_path)

    bi = building_info.copy()
    bi["사용그룹"] = bi["건물유형"].map(map_usage_group)

    group_version = {
        "24시간": "D",
        "기타": "B",
        "주간": "D"
    }

    all_preds = []

    for g in ["24시간", "기타", "주간"]:
        ver = group_version[g]
        bld_ids = bi.loc[bi["사용그룹"] == g, "건물번호"].unique()

        train_sub = train_raw[train_raw["건물번호"].isin(bld_ids)].copy()
        test_sub = test_raw[test_raw["건물번호"].isin(bld_ids)].copy()

        preds_df = train_and_predict_for_group(train_sub, test_sub, building_info, ver)
        all_preds.append(preds_df)

    all_preds_df = pd.concat(all_preds, ignore_index=True)

    submission = sample_sub.copy().drop(columns=["answer"], errors="ignore")
    submission = submission.merge(all_preds_df, on="num_date_time", how="left")
    submission["answer"].fillna(submission["answer"].mean(), inplace=True)

    submission.to_csv(output_path, index=False)
    print(f"\n✅ XGBoost 제출 파일 생성 완료: {output_path}")
