import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import holidays
import warnings

# ===========================
# 버전 설명
# ===========================
warnings.filterwarnings("ignore")

VERSION_DESC = {
    "A": "이상치 미적용",
    "B": "IQR + Z-score",
    "C": "규칙 기반 제거",
    "D": "규칙 + IQR + Z-score + Clipping"
}

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
# 전처리 클래스 (A버전 베이스)
# ===========================
class PowerPreprocessor:

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.kr_holidays = holidays.KR()
        self.scale_cols_ = None

    # 날짜/시간 파생
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

    # 건물정보 merge
    def merge_building_info(self, df, building_info):
        df = df.merge(building_info, on="건물번호", how="left")
        return df

    # 컬럼명 정규화 + 결측 처리
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

    # 범주형 인코딩
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

    # 파생변수 생성
    def create_features(self, df):
        df = df.copy()

        if "기온" in df.columns and "습도" in df.columns:
            df["THI"] = (
                9/5*df["기온"]
                - 0.55*(1-df["습도"]/100)*(9/5*df["기온"]-26)
                + 32
            )

        if "기온" in df.columns and "풍속" in df.columns:
            df["wind_chill"] = (
                13.12 + 0.6215*df["기온"] - 11.37*(df["풍속"]**0.16)
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

    # 스케일링
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

    # 전체 파이프라인
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
# 이상치 유틸 (B/C/D에서 사용)
# ===========================
def detect_missing_by_rules(df, name="Train"):
    df = df.copy()
    df["일시_dt"] = pd.to_datetime(df["일시"], errors="coerce")
    df["hour"] = df["일시_dt"].dt.hour

    rule_missing = pd.Series(False, index=df.index)

    if "기온(°C)" in df.columns:
        rule_missing |= (df["기온(°C)"] < -20) | (df["기온(°C)"] > 45)
    if "습도(%)" in df.columns:
        rule_missing |= (df["습도(%)"] < 1) | (df["습도(%)"] > 100)
    if "풍속(m/s)" in df.columns:
        rule_missing |= (df["풍속(m/s)"] < 0) | (df["풍속(m/s)"] > 25)
    if "강수량(mm)" in df.columns:
        rule_missing |= (df["강수량(mm)"] < 0) | (df["강수량(mm)"] > 200)
    if "일사(MJ/m2)" in df.columns:
        rule_missing |= (df["hour"] >= 20) & (df["일사(MJ/m2)"] > 0)

    if "전력소비량(kWh)" in df.columns:
        target_col = "전력소비량(kWh)"
    else:
        target_col = "전력소비량"

    if target_col in df.columns:
        rule_missing |= (df["hour"].between(8, 18)) & (df[target_col] == 0)

    for col in ["연면적(m2)", "냉방면적(m2)", "태양광용량(kW)"]:
        if col in df.columns:
            try:
                rule_missing |= df[col].astype(float) <= 0
            except:
                pass

    print(f"[{name}] 규칙 기반 플래그: {rule_missing.sum()}건")
    return rule_missing


def remove_outliers_iqr(df, col, factor=1.5):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    mask = (df[col] < lower) | (df[col] > upper)
    print(f"[IQR] {col} 기준 제거: {mask.sum()}건")
    return df[~mask].reset_index(drop=True)


def remove_outliers_zscore(df, col, threshold=3.0):
    mean = df[col].mean()
    std = df[col].std()
    if std == 0:
        print(f"[Z-score] {col} std=0 → 스킵")
        return df
    z = (df[col] - mean) / std
    mask = np.abs(z) > threshold
    print(f"[Z-score] {col} 기준 제거: {mask.sum()}건")
    return df[~mask].reset_index(drop=True)


def clip_target(df, col="전력소비량", lower_q=0.01, upper_q=0.99):
    low = df[col].quantile(lower_q)
    high = df[col].quantile(upper_q)
    print(f"[클리핑] {col}: [{low:.2f}, {high:.2f}]")
    df = df.copy()
    df[col] = df[col].clip(low, high)
    return df


# ===========================
# 3그룹 매핑 함수
# ===========================
def map_usage_group(btype):
    # 네가 말한 기준:
    #  - 24시간: 호텔, 병원, 백화점, 전화국(통신 관련)
    #  - 주간: 학교, 연구소
    #  - 나머지: 기타 (아파트, 공공, 상업 등)
    if btype in ["호텔", "병원", "백화점", "통신시설", "전화국"]:
        return "24시간"
    elif btype in ["학교", "연구소"]:
        return "주간"
    else:
        return "기타"


# ===========================
# LightGBM 평가
# ===========================
def evaluate_lgbm(train_processed, desc=""):
    df = train_processed.copy()

    if "전력소비량(kWh)" in df.columns:
        df = df.rename(columns={"전력소비량(kWh)": "전력소비량"})

    drop_cols = []
    for c in ["일시", "건물유형"]:
        if c in df.columns:
            drop_cols.append(c)

    X = df.drop(columns=["전력소비량"] + drop_cols)
    y = df["전력소비량"]

    X = X.select_dtypes(include=[np.number])

    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42
    )

    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_tr, y_tr)
    pred = model.predict(X_va)
    rmse = np.sqrt(mean_squared_error(y_va, pred))
    print(f"[{desc}] RMSE: {rmse:.4f}")
    return rmse


# ===========================
# 버전별 (전체) 실험
# ===========================
def run_version_experiment(version, train_raw, test_raw, building_info):
    print("\n" + "="*60)
    print(f"▶ 버전 {version} (전체 통합) 실험 시작")
    print("="*60)

    train = train_raw.copy()
    test = test_raw.copy()

    if "전력소비량(kWh)" in train.columns and "전력소비량" not in train.columns:
        train = train.rename(columns={"전력소비량(kWh)": "전력소비량"})

    if version == "A":
        pass

    elif version == "B":
        train = remove_outliers_iqr(train, "전력소비량", factor=1.5)
        train = remove_outliers_zscore(train, "전력소비량", threshold=3.0)

    elif version == "C":
        mask_rule = detect_missing_by_rules(train, name=f"Train-{version}")
        train = train[~mask_rule].reset_index(drop=True)

    elif version == "D":
        mask_rule = detect_missing_by_rules(train, name=f"Train-{version}")
        train = train[~mask_rule].reset_index(drop=True)
        train = remove_outliers_iqr(train, "전력소비량", factor=1.5)
        train = remove_outliers_zscore(train, "전력소비량", threshold=3.0)
        train = clip_target(train, col="전력소비량", lower_q=0.01, upper_q=0.99)
    else:
        raise ValueError("version must be one of ['A','B','C','D']")

    print(f"[{version}] 이상치 처리 후 train shape: {train.shape}")

    processor = PowerPreprocessor()
    train_processed, _ = processor.full_pipeline(train, test, building_info)

    rmse = evaluate_lgbm(train_processed, desc=f"버전 {version} (전체)")
    return rmse


# ===========================
# 버전별 + 3그룹별 실험
# ===========================
def run_version_experiment_by_usage_group(version, train_raw, test_raw, building_info):
    print("\n" + "="*60)
    print(f"▶ 버전 {version} (3그룹별) 실험 시작")
    print("="*60)

    results = {}

    # building_info에 3그룹 컬럼 추가
    bi = building_info.copy()
    bi["사용그룹"] = bi["건물유형"].map(map_usage_group)

    groups = bi["사용그룹"].dropna().unique()

    for g in groups:
        bld_ids = bi.loc[bi["사용그룹"] == g, "건물번호"].unique()

        train_sub = train_raw[train_raw["건물번호"].isin(bld_ids)].copy()
        if train_sub.empty:
            print(f"[{version}/{g}] 해당 그룹 데이터 없음 → 스킵")
            continue

        test = test_raw.copy()

        if "전력소비량(kWh)" in train_sub.columns and "전력소비량" not in train_sub.columns:
            train_sub = train_sub.rename(columns={"전력소비량(kWh)": "전력소비량"})

        # 버전별 이상치 처리 (그룹별 서브셋에 대해 적용)
        if version == "A":
            pass

        elif version == "B":
            train_sub = remove_outliers_iqr(train_sub, "전력소비량", factor=1.5)
            train_sub = remove_outliers_zscore(train_sub, "전력소비량", threshold=3.0)

        elif version == "C":
            mask_rule = detect_missing_by_rules(train_sub, name=f"{g}-Train-{version}")
            train_sub = train_sub[~mask_rule].reset_index(drop=True)

        elif version == "D":
            mask_rule = detect_missing_by_rules(train_sub, name=f"{g}-Train-{version}")
            train_sub = train_sub[~mask_rule].reset_index(drop=True)
            train_sub = remove_outliers_iqr(train_sub, "전력소비량", factor=1.5)
            train_sub = remove_outliers_zscore(train_sub, "전력소비량", threshold=3.0)
            train_sub = clip_target(train_sub, col="전력소비량", lower_q=0.01, upper_q=0.99)

        else:
            raise ValueError("version must be one of ['A','B','C','D']")

        if len(train_sub) < 100:
            print(f"[{version}/{g}] 데이터 {len(train_sub)}행 → 너무 적어서 스킵")
            continue

        print(f"[{version}/{g}] 이상치 처리 후 train_sub shape: {train_sub.shape}")

        processor = PowerPreprocessor()
        train_processed, _ = processor.full_pipeline(train_sub, test, building_info)

        rmse = evaluate_lgbm(
            train_processed,
            desc=f"버전 {version} / 사용그룹 {g}"
        )
        results[g] = rmse

    return results


# ===========================
# MAIN
# ===========================
if __name__ == "__main__":

    train_raw = load_csv_with_fallback("C:/data/train.csv")
    test_raw = load_csv_with_fallback("C:/data/test.csv")
    building_info = load_csv_with_fallback("C:/data/building_info.csv")

    # 타겟 컬럼 통일
    if "전력소비량(kWh)" in train_raw.columns and "전력소비량" not in train_raw.columns:
        train_raw = train_raw.rename(columns={"전력소비량(kWh)": "전력소비량"})

    # 1) 전체 통합 버전 A/B/C/D 비교
    total_results = {}
    for v in ["A", "B", "C", "D"]:
        rmse_v = run_version_experiment(v, train_raw, test_raw, building_info)
        total_results[v] = rmse_v

    print("\n" + "="*60)
    print("▶ 전체 통합 기준 버전별 RMSE")
    print("="*60)
    for v in ["A", "B", "C", "D"]:
        print(f"  버전 {v}: RMSE = {total_results[v]:.4f}")

    # 2) 3그룹별 버전 A/B/C/D 비교
    all_group_results = {}

    for v in ["A", "B", "C", "D"]:
        g_results = run_version_experiment_by_usage_group(
            v, train_raw, test_raw, building_info
        )
        all_group_results[v] = g_results

    print("\n" + "="*60)
    print("▶ 버전 / 사용그룹별 RMSE 요약")
    print("="*60)
    for v in ["A", "B", "C", "D"]:
        print(f"\n[버전 {v} | {VERSION_DESC[v]}]")
        if not all_group_results[v]:
            print("  결과 없음")
            continue
        for g, rmse in all_group_results[v].items():
            print(f"  사용그룹 {g}: RMSE = {rmse:.4f}")
