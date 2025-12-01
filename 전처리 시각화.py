import os
import sys
import io
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

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
# Windows 한글 폰트 설정
# ============================================================
def setup_korean_font_windows():
    try:
        fm._rebuild()
        print("✓ 폰트 캐시 재생성 완료")
    except:
        pass

    font_paths = [
        r"C:\Windows\Fonts\malgun.ttf",
        r"C:\Windows\Fonts\malgunbd.ttf",
        r"C:\Windows\Fonts\gulim.ttc",
        r"C:\Windows\Fonts\batang.ttc",
        r"C:\Windows\Fonts\NanumGothic.ttf",
    ]

    available_font = None
    for font_path in font_paths:
        if os.path.exists(font_path):
            available_font = font_path
            print(f"✓ 폰트 발견: {font_path}")
            break

    if available_font:
        font_prop = fm.FontProperties(fname=available_font)
        matplotlib.rcParams['font.family'] = font_prop.get_name()

    matplotlib.rcParams['font.sans-serif'] = ['Malgun Gothic', 'Gulim', 'Batang', 'NanumGothic']
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    print("✓ 한글 폰트 설정 완료 (Malgun Gothic)")

setup_korean_font_windows()

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

    # (1) 기온
    if "기온(°C)" in df.columns:
        cond = (df["기온(°C)"] < -20) | (df["기온(°C)"] > 45)
        print(f"[기온] 물리적 범위 벗어남: {cond.sum()}건")
        rule_missing |= cond

    # (2) 습도
    if "습도(%)" in df.columns:
        cond = (df["습도(%)"] < 1) | (df["습도(%)"] > 100)
        print(f"[습도] 비정상 값: {cond.sum()}건")
        rule_missing |= cond

    # (3) 풍속
    if "풍속(m/s)" in df.columns:
        cond = (df["풍속(m/s)"] < 0) | (df["풍속(m/s)"] > 25)
        print(f"[풍속] 비정상 값: {cond.sum()}건")
        rule_missing |= cond

    # (4) 야간 일사 오류
    if "일사(MJ/m2)" in df.columns:
        cond = (df["hour"] >= 20) & (df["일사(MJ/m2)"] > 0)
        print(f"[일사] 야간 일사값 오류: {cond.sum()}건")
        rule_missing |= cond

    # (5) 강수량
    if "강수량(mm)" in df.columns:
        cond = (df["강수량(mm)"] < 0) | (df["강수량(mm)"] > 200)
        print(f"[강수량] 비정상 값: {cond.sum()}건")
        rule_missing |= cond

    # (6) 운영시간 전력 0 이상치
    if ("전력소비량" in df.columns) and (not building_info):
        cond = (df["hour"].between(8, 18)) & (df["전력소비량"] == 0)
        print(f"[전력] 운영시간 전력 0: {cond.sum()}건")
        rule_missing |= cond

    # (7) 건물 정보
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
# IQR 기반 이상치 제거
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
# Z-score 기반 이상치 제거
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
        print("[경고] 건물유형 없음 → 스킵")
        return df

    cleaned = []
    for btype in df["건물유형"].unique():
        sub = df[df["건물유형"] == btype].copy()
        print(f"\n[건물유형: {btype}] count={len(sub)}")

        if btype in ["호텔", "병원"]:
            sub = remove_outliers_iqr(sub, "전력소비량", factor=2.0)
        elif btype in ["학교"]:
            sub = remove_outliers_iqr(sub, "전력소비량", factor=1.2)
        else:
            sub = remove_outliers_iqr(sub, "전력소비량", factor=1.5)

        cleaned.append(sub)

    result = pd.concat(cleaned).reset_index(drop=True)
    print("\n=== 건물유형별 이상치 규칙 적용 완료 === →", result.shape)
    return result

# ============================================================
# 전처리 이전 EDA (간단)
# ============================================================
def run_eda_before(train, building_info):
    sns.set(font_scale=1.2, font='Malgun Gothic')

    plt.figure(figsize=(10, 5))
    sns.histplot(train["전력소비량"], bins=80, kde=True)
    plt.title("전처리 이전: 전력 소비량 분포")
    plt.show()

# ============================================================
# 분포 KDE diff plot
# ============================================================
def plot_diff_before_after(before, after, col="전력소비량"):
    plt.figure(figsize=(12, 5))
    sns.kdeplot(before[col], label="Before", bw_adjust=1.3)
    sns.kdeplot(after[col], label="After", bw_adjust=1.3)
    plt.title(f"{col} 분포 비교: Before vs After")
    plt.legend()
    plt.grid(True)
    plt.xlabel(col)
    plt.show()

# ============================================================
# Boxplot Before vs After (raw)
# ============================================================
def plot_boxplot_compare(before, after, col="전력소비량"):
    plt.figure(figsize=(8, 5))
    data = pd.DataFrame({
        "value": pd.concat([before[col], after[col]], axis=0),
        "type": ["Before"] * len(before) + ["After"] * len(after)
    })
    sns.boxplot(data=data, x="type", y="value")
    plt.title(f"{col} Boxplot: Before vs After")
    plt.ylabel(col)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.show()

# ============================================================
# 건물유형별 Boxplot Before vs After
# ============================================================
def plot_building_type_boxplot(before, after):
    if "건물유형" not in before.columns:
        print("[건물유형 Boxplot] 건물유형 컬럼 없음 → 스킵")
        return

    df_before = before[["건물유형", "전력소비량"]].copy()
    df_before["type"] = "Before"

    df_after = after[["건물유형", "전력소비량"]].copy()
    df_after["type"] = "After"

    df = pd.concat([df_before, df_after], axis=0)

    plt.figure(figsize=(14, 6))
    sns.boxplot(data=df, x="건물유형", y="전력소비량", hue="type")
    plt.title("건물유형별 전력소비량 Boxplot: Before vs After")
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# ============================================================
# CDF(누적분포) 비교
# ============================================================
def plot_cdf_compare(before, after, col="전력소비량"):
    vb = np.sort(before[col].values)
    va = np.sort(after[col].values)
    cb = np.arange(1, len(vb)+1) / len(vb)
    ca = np.arange(1, len(va)+1) / len(va)

    plt.figure(figsize=(10, 5))
    plt.plot(vb, cb, label="Before")
    plt.plot(va, ca, label="After")
    plt.title(f"{col} CDF 비교: Before vs After")
    plt.xlabel(col)
    plt.ylabel("누적 비율")
    plt.grid(True)
    plt.legend()
    plt.show()

# ============================================================
# 상관관계 히트맵
# ============================================================
def plot_corr_heatmap(df, title="상관관계 히트맵"):
    cand_cols = [
        "전력소비량", "기온", "습도", "풍속", "강수량",
        "일조", "일사", "THI", "wind_chill",
        "cooling_ratio", "solar_density"
    ]
    cols = [c for c in cand_cols if c in df.columns]
    if len(cols) < 2:
        print("[상관 히트맵] 유효한 컬럼 부족 → 스킵")
        return

    corr = df[cols].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0, square=True,
                cbar_kws={"shrink": 0.7})
    plt.title(title)
    plt.tight_layout()
    plt.show()

# ============================================================
# 전처리 이후 EDA 비교 (Before vs After)
# ============================================================
def run_eda_compare(before_df, after_df):
    sns.set(font_scale=1.2, font='Malgun Gothic')

    # 1) 전력소비량 KDE 비교
    plt.figure(figsize=(12, 5))
    sns.kdeplot(before_df["전력소비량"], label="Before", bw_adjust=1.3)
    sns.kdeplot(after_df["전력소비량"], label="After", bw_adjust=1.3)
    plt.title("전력소비량 분포 비교 (전처리 이후)")
    plt.legend()
    plt.grid(True)
    plt.xlabel("전력소비량")
    plt.show()

    # 2) Boxplot 비교 (전처리 이후 스케일 그대로)
    plt.figure(figsize=(8, 5))
    data = pd.DataFrame({
        "value": pd.concat([before_df["전력소비량"], after_df["전력소비량"]], axis=0),
        "type": ["Before"] * len(before_df) + ["After"] * len(after_df)
    })
    sns.boxplot(data=data, x="type", y="value")
    plt.title("전처리 이후 전력소비량 Boxplot: Before vs After")
    plt.ylabel("전력소비량")
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.show()

    # 3) CDF 비교
    plot_cdf_compare(before_df, after_df, col="전력소비량")

    # 4) 시간대별 평균 전력소비량
    if "hour" in before_df.columns:
        plt.figure(figsize=(12, 5))
        plt.plot(before_df.groupby("hour")["전력소비량"].mean(), marker='o', label="Before")
        plt.plot(after_df.groupby("hour")["전력소비량"].mean(), marker='o', label="After")
        plt.title("시간대별 평균 전력소비량: Before vs After")
        plt.xlabel("hour")
        plt.ylabel("전력소비량")
        plt.legend()
        plt.grid(True)
        plt.show()

    # 5) 요일별 평균 전력소비량
    if "dayofweek" in before_df.columns:
        label_map = {0: "월", 1: "화", 2: "수", 3: "목", 4: "금", 5: "토", 6: "일"}
        bw = before_df.groupby("dayofweek")["전력소비량"].mean().reindex(range(7))
        aw = after_df.groupby("dayofweek")["전력소비량"].mean().reindex(range(7))

        plt.figure(figsize=(10, 5))
        plt.plot(bw.index, bw.values, marker='o', label="Before")
        plt.plot(aw.index, aw.values, marker='o', label="After")
        plt.xticks(range(7), [label_map[i] for i in range(7)])
        plt.title("요일별 평균 전력소비량: Before vs After")
        plt.ylabel("전력소비량")
        plt.grid(True)
        plt.legend()
        plt.show()

    # 6) THI vs 전력소비량
    if "THI" in before_df.columns:
        plt.figure(figsize=(10, 5))
        sns.scatterplot(
            data=before_df.sample(min(2000, len(before_df)), random_state=42),
            x="THI", y="전력소비량", alpha=0.4, label="Before"
        )
        sns.scatterplot(
            data=after_df.sample(min(2000, len(after_df)), random_state=42),
            x="THI", y="전력소비량", alpha=0.4, label="After"
        )
        plt.title("THI vs 전력소비량 비교")
        plt.grid(True)
        plt.legend()
        plt.show()

    # 7) wind_chill vs 전력소비량
    if "wind_chill" in before_df.columns:
        plt.figure(figsize=(10, 5))
        sns.scatterplot(
            data=before_df.sample(min(2000, len(before_df)), random_state=42),
            x="wind_chill", y="전력소비량", alpha=0.4, label="Before"
        )
        sns.scatterplot(
            data=after_df.sample(min(2000, len(after_df)), random_state=42),
            x="wind_chill", y="전력소비량", alpha=0.4, label="After"
        )
        plt.title("체감온도 vs 전력소비량 비교")
        plt.grid(True)
        plt.legend()
        plt.show()

    # 8) 기온 vs 전력소비량 회귀선
    if "기온" in before_df.columns:
        plt.figure(figsize=(10, 5))
        sns.regplot(
            data=before_df.sample(min(5000, len(before_df)), random_state=42),
            x="기온", y="전력소비량", scatter_kws={"alpha": 0.3},
            label="Before", line_kws={"color": "C0"}
        )
        sns.regplot(
            data=after_df.sample(min(5000, len(after_df)), random_state=42),
            x="기온", y="전력소비량", scatter_kws={"alpha": 0.3},
            label="After", line_kws={"color": "C1"}
        )
        plt.title("기온 vs 전력소비량 회귀 관계: Before vs After")
        plt.grid(True)
        plt.legend()
        plt.show()

# ============================================================
# 전처리 파이프라인
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
            if m in [12, 1, 2]: return 0
            if m in [3, 4, 5]: return 1
            if m in [6, 7, 8]: return 2
            return 3
        df["season"] = df["month"].apply(season)

        def time_of_day(h):
            if 6 <= h < 12: return 0
            if 12 <= h < 18: return 1
            if 18 <= h < 22: return 2
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
# 전처리 실행 함수
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
# Feature Importance 플롯
# ============================================================
def plot_feature_importance(model, feature_names, title="Feature Importance", top_n=20):
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]
    plt.figure(figsize=(8, max(4, top_n * 0.3)))
    plt.barh(range(len(idx)), importances[idx][::-1])
    plt.yticks(range(len(idx)), [feature_names[i] for i in idx][::-1])
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()
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
    return rmse, model, X.columns

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    print("\n" + "="*60)
    print("전력 데이터 전처리 + 이상치 제거 + 전후 비교 EDA + 모델 성능 비교")
    print("="*60)

    # 1. 데이터 로드
    train_raw = load_csv_with_fallback("C:/data/train.csv")
    test_raw = load_csv_with_fallback("C:/data/test.csv")
    building_info = load_csv_with_fallback("C:/data/building_info.csv")

    # 타겟 이름 통일
    if "전력소비량(kWh)" in train_raw.columns:
        train_raw = train_raw.rename(columns={"전력소비량(kWh)": "전력소비량"})

    # 2. 전처리 이전 EDA(간단)
    print("\n[1] 전처리 이전 EDA (간단)")
    run_eda_before(train_raw, building_info)

    # 3. 규칙 기반 이상치 제거
    print("\n[2] 규칙 기반 이상치 제거")
    rule_mask_train = detect_missing_by_rules(train_raw, False, "Train")
    rule_mask_test = detect_missing_by_rules(test_raw, False, "Test")

    train_rule_clean = train_raw[~rule_mask_train].reset_index(drop=True)
    test_rule_clean = test_raw[~rule_mask_test].reset_index(drop=True)

    # 4. IQR + Z-score 제거
    print("\n[3] IQR + Z-score 제거")
    train_iqr_clean = remove_outliers_iqr(train_rule_clean, "전력소비량")
    train_z_clean = remove_outliers_zscore(train_iqr_clean, "전력소비량")

    # 5. 건물유형별 규칙 적용
    print("\n[4] 건물유형별 이상치 제거")
    train_final_clean = remove_outliers_by_building_type(train_z_clean)

    # 6. 분포 / Boxplot / CDF / 건물유형 Boxplot 비교 (raw 기준)
    print("\n[5] 전력소비량 분포 KDE 비교 (raw)")
    plot_diff_before_after(train_raw, train_final_clean, col="전력소비량")

    print("\n[5-1] 전력소비량 Boxplot 비교 (raw)")
    plot_boxplot_compare(train_raw, train_final_clean, col="전력소비량")

    print("\n[5-2] 전력소비량 CDF 비교 (raw)")
    plot_cdf_compare(train_raw, train_final_clean, col="전력소비량")

    print("\n[5-3] 건물유형별 Boxplot 비교 (raw)")
    plot_building_type_boxplot(train_raw, train_final_clean)

    # 7. 전처리 Before / After
    print("\n[6] 전처리 실행 (Before 데이터)")
    train_processed_before, _ = preprocess_loaded_data(train_raw, test_raw, building_info)

    print("\n[7] 전처리 실행 (After 데이터)")
    train_processed_after, test_processed = preprocess_loaded_data(train_final_clean, test_rule_clean, building_info)

    # 8. 전처리 이후 EDA 비교
    print("\n[8] 전처리 이후 EDA 비교 (Before vs After)")
    run_eda_compare(train_processed_before, train_processed_after)

    print("\n[8-1] 상관관계 히트맵 (Before)")
    plot_corr_heatmap(train_processed_before, "상관관계 히트맵 (Before)")

    print("\n[8-2] 상관관계 히트맵 (After)")
    plot_corr_heatmap(train_processed_after, "상관관계 히트맵 (After)")

    # 9. 모델 성능 비교
    print("\n[9] 모델 성능 비교 (LightGBM)")
    rmse_before, model_before, feat_names = evaluate_model(train_processed_before, "이상치 제거 전")
    rmse_after, model_after, _ = evaluate_model(train_processed_after, "이상치 제거 후")

    print("\n===== 최종 RMSE 비교 =====")
    print(f"이상치 제거 전 RMSE : {rmse_before:.4f}")
    print(f"이상치 제거 후 RMSE : {rmse_after:.4f}")

    # 10. Feature Importance 비교
    print("\n[10] Feature Importance (Before)")
    plot_feature_importance(model_before, feat_names, "Feature Importance (Before)")

    print("\n[11] Feature Importance (After)")
    plot_feature_importance(model_after, feat_names, "Feature Importance (After)")

    print("\n===== 모든 작업 완료 =====")
