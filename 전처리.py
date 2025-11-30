import os
import sys
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder

#데이터 한글 깨짐 해결 코드
os.environ.setdefault('PYTHONUTF8', '1')
try:
	# Python 3.7+ supports reconfigure on TextIOBase
	sys.stdout.reconfigure(encoding='utf-8')
	sys.stderr.reconfigure(encoding='utf-8')
except Exception:
	# Fallback for older versions / environments
	try:
		sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
		sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
	except Exception:
		pass

#한글 깨짐 해결 코드
def load_csv_with_fallback(path):
	"""Try several encodings and return the first successful DataFrame.

	Tries: utf-8-sig, utf-8, cp949, euc-kr. Logs which encoding succeeded.
	"""
	encodings = ['utf-8-sig', 'utf-8', 'cp949', 'euc-kr']
	for enc in encodings:
		try:
			df = pd.read_csv(path, encoding=enc)
			print(f"Loaded '{path}' with encoding: {enc}")
			return df
		except Exception:
			continue
	# Last resort: let pandas read with latin1 to avoid failure (may mangle characters)
	try:
		df = pd.read_csv(path, encoding='latin1')
		print(f"Loaded '{path}' with fallback encoding: latin1")
		return df
	except Exception as e:
		raise

#데이터 불러오기
test = load_csv_with_fallback(r'C:/Users/daaak/OneDrive/문서/바탕 화면/data/test.csv')
train = load_csv_with_fallback(r'C:/Users/daaak/OneDrive/문서/바탕 화면/data/train.csv')
building_info = load_csv_with_fallback(r'C:/Users/daaak/OneDrive/문서/바탕 화면/data/building_info.csv')

# ============================================================
# UTF-8 인코딩 강제 설정
# ============================================================
os.environ.setdefault('PYTHONUTF8', '1')
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass



# ============================================================
# CSV 로드 함수
# ============================================================
def load_csv_with_fallback(path):
    encodings = ['utf-8-sig', 'utf-8', 'cp949', 'euc-kr']
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"✓ '{path}' 로드 완료 (인코딩: {enc})")
            return df
        except:
            continue
    df = pd.read_csv(path, encoding='latin1')
    print(f"✓ '{path}' 로드 완료 (fallback: latin1)")
    return df

# ============================================================
# 규칙 기반 결측치/이상치 탐지 함수
# ============================================================
def detect_missing_by_rules(df, building_info=False, name="Train"):

    print("\n" + "="*60)
    print(f" 규칙 기반 결측치/이상치 탐지 결과: {name}")
    print("="*60)

    df = df.copy()
    df["일시_dt"] = pd.to_datetime(df["일시"], errors="coerce")
    df["hour"] = df["일시_dt"].dt.hour

    rule_missing = pd.Series(False, index=df.index)

    # 1) 기온
    if "기온(°C)" in df.columns:
        cond = (df["기온(°C)"] < -20) | (df["기온(°C)"] > 45)
        print(f"[기온] 물리적 범위 벗어남: {cond.sum()}건")
        rule_missing |= cond

    # 2) 습도
    if "습도(%)" in df.columns:
        cond = (df["습도(%)"] < 1) | (df["습도(%)"] > 100)
        print(f"[습도] 비정상 값: {cond.sum()}건")
        rule_missing |= cond

    # 3) 풍속
    if "풍속(m/s)" in df.columns:
        cond = (df["풍속(m/s)"] < 0) | (df["풍속(m/s)"] > 25)
        print(f"[풍속] 비정상 값: {cond.sum()}건")
        rule_missing |= cond

    # 4) 일사야간 오류
    if "일사(MJ/m2)" in df.columns:
        cond = (df["hour"] >= 20) & (df["일사(MJ/m2)"] > 0)
        print(f"[일사] 야간 일사값 오류: {cond.sum()}건")
        rule_missing |= cond

    # 5) 강수
    if "강수량(mm)" in df.columns:
        cond = (df["강수량(mm)"] < 0) | (df["강수량(mm)"] > 200)
        print(f"[강수량] 비정상 값: {cond.sum()}건")
        rule_missing |= cond

    # 6) 전력(Train Only)
    if ("전력소비량" in df.columns) and (not building_info):
        cond = (df["hour"].between(8, 18)) & (df["전력소비량"] == 0)
        print(f"[전력] 운영시간 전력 0: {cond.sum()}건")
        rule_missing |= cond

    # 7) 건물 정보
    for col in ["연면적(m2)", "냉방면적(m2)", "태양광용량(kW)"]:
        if col in df.columns:
            try:
                cond = df[col].astype(float) <= 0
                print(f"[건물정보] {col} <= 0: {cond.sum()}건")
                rule_missing |= cond
            except:
                pass

    print("\n총 규칙 기반 결측치/이상치:", rule_missing.sum(), "건")
    return rule_missing

# ============================================================
# IQR 기반 이상치 제거 함수
# ============================================================
def remove_outliers_iqr(df, col, factor=1.5):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    mask = (df[col] < lower) | (df[col] > upper)
    print(f"[IQR 제거] {col} 기준 이상치: {mask.sum()}건 제거")
    return df[~mask].reset_index(drop=True)

# ============================================================
# Z-score 기반 이상치 제거 함수
# ============================================================
def remove_outliers_zscore(df, col, threshold=3.0):
    mean = df[col].mean()
    std = df[col].std()
    if std == 0:
        print(f"[Z-score 제거] {col} 표준편차 0 → 스킵")
        return df
    z = (df[col] - mean) / std
    mask = np.abs(z) > threshold
    print(f"[Z-score 제거] {col} 기준 이상치: {mask.sum()}건 제거")
    return df[~mask].reset_index(drop=True)

# ============================================================
# 건물유형별 이상치 규칙 차등 적용
# ============================================================
def remove_outliers_by_building_type(train):
    print("\n=== 건물유형별 이상치 규칙 적용 시작 ===")

    df = train.copy()
    if "건물유형" not in df.columns:
        print("[경고] 건물유형 컬럼이 없어 건물유형별 이상치 제거를 수행할 수 없음.")
        return df

    unique_types = df["건물유형"].unique()
    cleaned_list = []

    for btype in unique_types:
        sub = df[df["건물유형"] == btype].copy()
        print(f"\n[건물유형: {btype}] 시작, 샘플 수: {len(sub)}")

        if btype in ["호텔", "병원"]:
            # 변동 큰 업종 → 느슨한 기준
            sub = remove_outliers_iqr(sub, "전력소비량", factor=2.0)
        elif btype in ["학교"]:
            # 학교는 전력 변동이 비교적 작음 → 엄격한 기준
            sub = remove_outliers_iqr(sub, "전력소비량", factor=1.2)
        else:
            # 기본 규칙
            sub = remove_outliers_iqr(sub, "전력소비량", factor=1.5)

        cleaned_list.append(sub)

    result = pd.concat(cleaned_list, axis=0).reset_index(drop=True)
    print("\n=== 건물유형별 이상치 규칙 적용 완료 ===")
    print("총 제거 후 train shape:", result.shape)

    return result

# ============================================================
# 전처리 이전 EDA
# ============================================================
def run_eda_before(train, building_info):
    sns.set(font_scale=1.2, font='Malgun Gothic')
    print("\n===== 전처리 이전 EDA 시작 =====")

    plt.figure(figsize=(10,5))
    sns.histplot(train["전력소비량"], bins=80, kde=True)
    plt.title("전처리 이전: 전력 소비량 분포")
    plt.show()

    plt.figure(figsize=(10,5))
    sns.histplot(train["기온(°C)"], bins=60, kde=True)
    plt.title("전처리 이전: 기온 분포")
    plt.show()

    plt.figure(figsize=(10,5))
    sns.histplot(train["습도(%)"], bins=60, kde=True)
    plt.title("전처리 이전: 습도 분포")
    plt.show()

    if "건물유형" in building_info.columns:
        plt.figure(figsize=(10,5))
        sns.countplot(data=building_info, x="건물유형")
        plt.title("전처리 이전: 건물 유형 분포")
        plt.xticks(rotation=45)
        plt.show()

# ============================================================
# 이상치 제거 전/후 diff plot
# ============================================================
def plot_diff_before_after(before, after, col="전력소비량"):
    plt.figure(figsize=(12,5))
    sns.kdeplot(before[col], label="Before", bw_adjust=1.5)
    sns.kdeplot(after[col], label="After", bw_adjust=1.5)
    plt.title(f"{col} 분포 비교: 제거 전 vs 제거 후")
    plt.legend()
    plt.grid(True)
    plt.show()

# ============================================================
# 전처리 클래스
# ============================================================
class PowerDataPreprocessor:

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    def preprocess_datetime(self, df):
        df = df.copy()
        df["일시"] = pd.to_datetime(df["일시"])
        df["year"] = df["일시"].dt.year
        df["month"] = df["일시"].dt.month
        df["day"] = df["일시"].dt.day
        df["hour"] = df["일시"].dt.hour
        df["dayofweek"] = df["일시"].dt.dayofweek
        df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

        def season(m):
            if m in [12, 1, 2]:
                return 0
            if m in [3, 4, 5]:
                return 1
            if m in [6, 7, 8]:
                return 2
            return 3
        df["season"] = df["month"].apply(season)

        def time_of_day(h):
            if 6 <= h < 12:
                return 0
            if 12 <= h < 18:
                return 1
            if 18 <= h < 22:
                return 2
            return 3
        df["time_of_day"] = df["hour"].apply(time_of_day)

        return df

    def merge_building_info(self, df, building_info):
        df = df.merge(building_info, on="건물번호", how="left")
        if "건물유형" in df.columns:
            df["건물유형_encoded"] = self.label_encoder.fit_transform(df["건물유형"])
        return df

    def handle_missing_values(self, df):
        df = df.copy()
        df = df.rename(columns={
            "기온(°C)": "기온",
            "습도(%)": "습도",
            "풍속(m/s)": "풍속",
            "강수량(mm)": "강수량",
            "일조(hr)": "일조",
            "일사(MJ/m2)": "일사"
        })

        weather_cols = ["기온", "강수량", "풍속", "습도", "일조", "일사"]
        for col in weather_cols:
            if col in df.columns:
                df[col] = df.groupby("건물번호")[col].ffill().bfill()

        building_cols = ["연면적(m2)", "냉방면적(m2)", "태양광용량(kW)", "ESS저장용량(kWh)", "PCS용량(kW)"]
        for col in building_cols:
            if col in df.columns:
                df[col] = df[col].replace("-", np.nan)
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col].fillna(df[col].median(), inplace=True)

        return df

    def create_features(self, df):
        df = df.copy()

        if "기온" in df.columns and "습도" in df.columns:
            df["THI"] = (
                9 / 5 * df["기온"]
                - 0.55 * (1 - df["습도"] / 100) * (9 / 5 * df["기온"] - 26)
                + 32
            )

        if "기온" in df.columns and "풍속" in df.columns:
            df["wind_chill"] = (
                13.12 + 0.6215 * df["기온"]
                - 11.37 * (df["풍속"] ** 0.16)
                + 0.3965 * df["기온"] * (df["풍속"] ** 0.16)
            )

        if "연면적(m2)" in df.columns and "냉방면적(m2)" in df.columns:
            df["cooling_ratio"] = df["냉방면적(m2)"] / (df["연면적(m2)"] + 1)

        if "태양광용량(kW)" in df.columns and "연면적(m2)" in df.columns:
            df["solar_density"] = df["태양광용량(kW)"] / (df["연면적(m2)"] + 1)

        return df

    def scale_features(self, train_df, test_df):
        train_df = train_df.copy()
        test_df = test_df.copy()

        numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
        exclude = [
            "전력소비량", "건물번호", "year", "month", "day",
            "hour", "dayofweek", "is_weekend", "season",
            "time_of_day", "건물유형_encoded"
        ]

        scale_cols = [c for c in numeric_cols if c not in exclude and c in test_df.columns]

        if scale_cols:
            self.scaler.fit(train_df[scale_cols])
            train_df[scale_cols] = self.scaler.transform(train_df[scale_cols])
            test_df[scale_cols] = self.scaler.transform(test_df[scale_cols])

        return train_df, test_df

# ============================================================
# 전처리 전체 실행
# ============================================================
def preprocess_loaded_data(train, test, building_info):
    print("\n===== 전처리 시작 =====")
    pre = PowerDataPreprocessor()

    train_df = pre.preprocess_datetime(train)
    test_df = pre.preprocess_datetime(test)

    train_df = pre.merge_building_info(train_df, building_info)
    test_df = pre.merge_building_info(test_df, building_info)

    train_df = pre.handle_missing_values(train_df)
    test_df = pre.handle_missing_values(test_df)

    train_df = pre.create_features(train_df)
    test_df = pre.create_features(test_df)

    train_processed, test_processed = pre.scale_features(train_df, test_df)

    print("✓ 전처리 완료")
    return train_processed, test_processed

# ============================================================
# 전처리 후 EDA
# ============================================================
def run_eda_after(train_df, title_prefix=""):

    plt.figure(figsize=(10, 5))
    sns.histplot(train_df["전력소비량"], bins=80, kde=True)
    plt.title(f"{title_prefix}전처리 이후: 전력소비량 분포")
    plt.show()

    if "hour" in train_df.columns:
        plt.figure(figsize=(12, 5))
        plt.plot(train_df.groupby("hour")["전력소비량"].mean(), marker='o')
        plt.title(f"{title_prefix}전처리 이후: 시간대별 평균 전력소비량")
        plt.grid(True)
        plt.show()

    if "THI" in train_df.columns:
        sample = train_df.sample(min(3000, len(train_df)), random_state=42)
        plt.figure(figsize=(10, 5))
        sns.scatterplot(data=sample, x="THI", y="전력소비량")
        plt.title(f"{title_prefix}전처리 이후: THI vs 전력소비량")
        plt.show()

    if "wind_chill" in train_df.columns:
        sample = train_df.sample(min(3000, len(train_df)), random_state=42)
        plt.figure(figsize=(10, 5))
        sns.scatterplot(data=sample, x="wind_chill", y="전력소비량")
        plt.title(f"{title_prefix}전처리 이후: 체감온도 vs 전력소비량")
        plt.show()

# ============================================================
# 모델 성능 평가 (LightGBM)
# ============================================================
def evaluate_model(train_df, desc=""):
    df = train_df.copy()

    drop_cols = ["일시", "num_date_time", "건물유형"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    X = df.drop(columns=["전력소비량"])
    y = df["전력소비량"]

    X_train, X_valid, y_train, y_valid = train_test_split(
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

    model.fit(X_train, y_train)
    pred = model.predict(X_valid)
    rmse = np.sqrt(mean_squared_error(y_valid, pred))
    print(f"[{desc}] RMSE: {rmse:.4f}")

    return rmse

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    print("\n" + "="*60)
    print("전력 데이터 전처리 + 이상치 탐지/제거 + 전처리 후 그래프 + 성능 비교")
    print("="*60)

    # 1. 데이터 로드
    train_raw = load_csv_with_fallback("C:/data/train.csv")
    test_raw = load_csv_with_fallback("C:/data/test.csv")
    building_info = load_csv_with_fallback("C:/data/building_info.csv")

    # 타겟 컬럼 이름 통일
    if "전력소비량(kWh)" in train_raw.columns:
        train_raw = train_raw.rename(columns={"전력소비량(kWh)": "전력소비량"})

    # 2. 전처리 이전 EDA
    print("\n[1] 전처리 이전 EDA")
    run_eda_before(train_raw, building_info)

    # 3. 규칙 기반 이상치 탐지 및 제거
    print("\n[2] 규칙 기반 이상치 탐지 및 제거")
    rule_mask_train = detect_missing_by_rules(train_raw, False, "Train")
    rule_mask_test = detect_missing_by_rules(test_raw, False, "Test")

    train_rule_clean = train_raw[~rule_mask_train].reset_index(drop=True)
    test_rule_clean = test_raw[~rule_mask_test].reset_index(drop=True)
    print(f"[규칙 기반 제거] Train: {train_raw.shape} → {train_rule_clean.shape}")
    print(f"[규칙 기반 제거] Test : {test_raw.shape} → {test_rule_clean.shape}")

    # 4. IQR + Z-score 기반 추가 이상치 제거 (전력소비량 기준)
    print("\n[3] IQR + Z-score 기반 이상치 제거 (전력소비량 기준)")
    train_iqr_clean = remove_outliers_iqr(train_rule_clean, "전력소비량", factor=1.5)
    train_z_clean = remove_outliers_zscore(train_iqr_clean, "전력소비량", threshold=3.0)
    print(f"[IQR+Z 제거] Train: {train_rule_clean.shape} → {train_z_clean.shape}")

    # 5. 건물유형별 이상치 규칙 차등 적용
    print("\n[4] 건물유형별 이상치 규칙 차등 적용")
    train_final_clean = remove_outliers_by_building_type(train_z_clean)
    print(f"[건물유형별 제거 후] Train: {train_z_clean.shape} → {train_final_clean.shape}")

    # 6. 이상치 제거 전/후 분포 비교
    print("\n[5] 이상치 제거 전/후 분포 비교 (diff plot)")
    plot_diff_before_after(train_raw, train_final_clean, col="전력소비량")

    # 7. 전처리 (제거 전 / 제거 후 모두 수행하여 비교)
    print("\n[6] 전처리 실행 (이상치 제거 전 데이터)")
    train_processed_before, _ = preprocess_loaded_data(train_raw, test_raw, building_info)

    print("\n[7] 전처리 실행 (이상치 제거 후 데이터)")
    train_processed_after, test_processed = preprocess_loaded_data(train_final_clean, test_rule_clean, building_info)

    # 8. 전처리 이후 EDA (전/후 비교)
    print("\n[8] 전처리 이후 EDA - 이상치 제거 전")
    run_eda_after(train_processed_before, title_prefix="[제거 전] ")

    print("\n[9] 전처리 이후 EDA - 이상치 제거 후")
    run_eda_after(train_processed_after, title_prefix="[제거 후] ")

    # 9. 모델 성능 비교
    print("\n[10] 모델 성능 비교 (LightGBM 기준)")
    rmse_before = evaluate_model(train_processed_before, desc="이상치 제거 전")
    rmse_after = evaluate_model(train_processed_after, desc="이상치 제거 후")

    print("\n===== 최종 RMSE 비교 =====")
    print(f"이상치 제거 전 RMSE: {rmse_before:.4f}")
    print(f"이상치 제거 후 RMSE: {rmse_after:.4f}")

    print("\n===== 모든 작업 완료 =====")
