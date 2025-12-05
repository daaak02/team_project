import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
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
            print(f"로드 완료: {path}")
            return df
        except:
            pass
    return pd.read_csv(path, encoding="latin1")

# ===========================
# 전처리 클래스 (전처리 전용 최종본)
# ===========================
class PowerPreprocessor:

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.kr_holidays = holidays.KR()
        self.scale_cols_ = None

    #날짜/시간 파생
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

    #건물정보 merge
    def merge_building_info(self, df, building_info):
        df = df.merge(building_info, on="건물번호", how="left")
        return df

    #컬럼명 정규화 + 결측 처리
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
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df.groupby("건물번호")[col].ffill().bfill()

        building_cols = ["연면적(m2)", "냉방면적(m2)", "태양광용량(kW)",
                         "ESS저장용량(kWh)", "PCS용량(kW)"]
        for col in building_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col].fillna(df[col].median(), inplace=True)

        return df

    #범주형 인코딩 (train 기준)
    def encode_categoricals(self, train_df, test_df, cat_cols):

        train_df = train_df.copy()
        test_df = test_df.copy()

        for col in cat_cols:
            if col not in self.label_encoders:
                le = LabelEncoder()
                train_df[col] = train_df[col].astype(str)
                test_df[col] = test_df[col].astype(str)

                le.fit(train_df[col])
                self.label_encoders[col] = le

            le = self.label_encoders[col]
            train_df[col] = le.transform(train_df[col])
            test_df[col] = test_df[col].map(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )

        return train_df, test_df

    #파생변수 생성
    def create_features(self, df):
        df = df.copy()

        if "기온" in df.columns and "습도" in df.columns:
            df["THI"] = 9/5*df["기온"] - 0.55*(1-df["습도"]/100)*(9/5*df["기온"]-26)+32

        if "기온" in df.columns and "풍속" in df.columns:
            df["wind_chill"] = 13.12 + 0.6215*df["기온"] - 11.37*(df["풍속"]**0.16)

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

    #스케일링 (train 기준, test 자동정렬)
    def scale(self, train_df, test_df):

        exclude = ["전력소비량", "건물번호", "일시", "num_date_time",
                   "year", "month", "day", "hour",
                   "dayofweek", "is_weekend", "is_holiday"]

        num_cols = train_df.select_dtypes(include=[np.number]).columns
        self.scale_cols_ = [c for c in num_cols if c not in exclude]

        self.scaler.fit(train_df[self.scale_cols_])
        train_df[self.scale_cols_] = self.scaler.transform(train_df[self.scale_cols_])

        for col in self.scale_cols_:
            if col not in test_df.columns:
                test_df[col] = 0

        test_df[self.scale_cols_] = self.scaler.transform(test_df[self.scale_cols_])

        return train_df, test_df

    #전체 파이프라인
    def full_pipeline(self, train, test, building_info):

        train = self.preprocess_datetime(train)
        test = self.preprocess_datetime(test)

        train = self.merge_building_info(train, building_info)
        test = self.merge_building_info(test, building_info)

        train = self.handle_missing_and_rename(train)
        test = self.handle_missing_and_rename(test)

        cat_cols = [c for c in ["건물유형"] if c in train.columns]
        train, test = self.encode_categoricals(train, test, cat_cols)

        train = self.create_features(train)
        test = self.create_features(test)

        train, test = self.scale(train, test)

        return train, test

# ===========================
# MAIN (전처리만 수행)
# ===========================
if __name__ == "__main__":

    train = load_csv_with_fallback("C:/data/train.csv")
    test = load_csv_with_fallback("C:/data/test.csv")
    building_info = load_csv_with_fallback("C:/data/building_info.csv")

    train = train.rename(columns={"전력소비량(kWh)": "전력소비량"})

    processor = PowerPreprocessor()
    train_processed, test_processed = processor.full_pipeline(
        train, test, building_info
    )

    train_processed.to_csv("C:/data/train_processed.csv", index=False)
    test_processed.to_csv("C:/data/test_processed.csv", index=False)

    print("\n 전처리 완료")
    print(" C:/data/train_processed.csv")
    print(" C:/data/test_processed.csv 생성")
