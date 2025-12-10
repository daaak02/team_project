import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import holidays
import warnings
warnings.filterwarnings("ignore")

# ===========================
# 1. CSV 로드
# ===========================
def load_csv_with_fallback(path):
    encodings = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"✓ 로드 완료: {path} ({enc})")
            return df
        except Exception:
            pass
    print(f"※ fallback(latin1) 로드: {path}")
    return pd.read_csv(path, encoding="latin1")


# ===========================
# 1-1. 일조/일사 강제 통일
# ===========================
def force_weather_columns(df):
    rename_map = {
        "일조(hr)": "일조",
        "일사(MJ/m2)": "일사"
    }
    for old, new in rename_map.items():
        if old in df.columns:
            df = df.rename(columns={old: new})

    for col in ["일조", "일사"]:
        if col not in df.columns:
            df[col] = np.nan

    return df


# ===========================
#   사용그룹 매핑
#   24시간: 호텔, 병원, 통신시설, 전화국
#   주간  : 학교, 연구소, 백화점
#   기타  : 나머지
# ===========================
def map_usage_group(btype):
    if btype in ["호텔", "병원", "통신시설", "전화국"]:
        return "24시간"
    elif btype in ["학교", "연구소", "백화점"]:
        return "주간"
    else:
        return "기타"


# ===========================
# 2. 전처리 + 시계열 클래스
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

        # 기상 컬럼 강제 생성 + 보간
        for col in ["기온", "강수량", "풍속", "습도", "일조", "일사"]:
            if col not in df.columns:
                df[col] = np.nan
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df.groupby("건물번호")[col].ffill().bfill()

        # 건물 고정 변수
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

        df["THI"] = (
            9/5 * df["기온"]
            - 0.55*(1-df["습도"]/100)*(9/5*df["기온"]-26)
            + 32
        )
        df["wind_chill"] = 13.12 + 0.6215*df["기온"] - 11.37*(df["풍속"]**0.16)
        df["cooling_ratio"] = df["냉방면적(m2)"] / (df["연면적(m2)"] + 1)
        df["solar_density"] = df["태양광용량(kW)"] / (df["연면적(m2)"] + 1)
        df["temp_sq"] = df["기온"] ** 2

        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        df["solar_effect"] = df["일사"] * df["태양광용량(kW)"]

        return df

    def add_lag_features(self, df):
        """
        전력소비량 lag(1,24,168) - train에서만 사용.
        """
        df = df.copy()
        if "전력소비량" not in df.columns:
            return df
        for lag in [1, 24, 168]:
            df[f"lag_{lag}"] = df.groupby("건물번호")["전력소비량"].shift(lag)
        return df

    def full_pipeline_train(self, train, building_info):
        # 여러 건물 포함한 train 전체에 대해 전처리 + lag 생성
        train = self.preprocess_datetime(train)
        train = self.merge_building_info(train, building_info)
        train = self.handle_missing_and_rename(train)
        train = self.create_features(train)
        train = self.add_lag_features(train)
        return train

    def full_pipeline_test_step(self, cur_row, building_info):
        # test 한 row에 대해 전처리 (lag는 밖에서 hist로 채움)
        df = cur_row.copy()
        df = self.preprocess_datetime(df)
        df = self.merge_building_info(df, building_info)
        df = self.handle_missing_and_rename(df)
        df = self.create_features(df)
        return df


# ===========================
# 3. 사용그룹별 / 건물별 모델 학습
# ===========================
def train_group_model_xgb(train_raw, building_info, bld_ids):
    """
    하나의 사용그룹(여러 건물 포함)에 대한 공통 XGBoost 모델 학습.
    """
    train_sub = train_raw[train_raw["건물번호"].isin(bld_ids)].copy()

    if len(train_sub) < 1000:
        return None, None, None

    proc = PowerPreprocessorXGB()
    train_feat = proc.full_pipeline_train(train_sub, building_info)

    if "lag_168" in train_feat.columns:
        train_feat = train_feat.dropna(subset=["lag_168"]).reset_index(drop=True)

    drop_cols = [c for c in ["일시", "num_date_time"] if c in train_feat.columns]
    X_train = train_feat.drop(columns=["전력소비량"] + drop_cols, errors="ignore")
    X_train = X_train.select_dtypes(include=[np.number])
    y_train = np.log1p(train_feat["전력소비량"])

    feature_cols = X_train.columns.tolist()

    model = XGBRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=8,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model, proc, feature_cols


def train_building_model_xgb(train_raw, building_info, bld_id):
    """
    한 건물에 대한 개별 XGBoost 모델 학습.
    (기존 15점 코드 구조와 동일)
    """
    train_sub = train_raw[train_raw["건물번호"] == bld_id].copy()

    if len(train_sub) < 500:
        return None, None, None, None

    proc = PowerPreprocessorXGB()
    train_feat = proc.full_pipeline_train(train_sub, building_info)

    if "lag_168" in train_feat.columns:
        train_feat = train_feat.dropna(subset=["lag_168"]).reset_index(drop=True)

    drop_cols = [c for c in ["일시", "num_date_time"] if c in train_feat.columns]
    X_train = train_feat.drop(columns=["전력소비량"] + drop_cols, errors="ignore")
    X_train = X_train.select_dtypes(include=[np.number])
    y_train = np.log1p(train_feat["전력소비량"])

    feature_cols = X_train.columns.tolist()

    model = XGBRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=8,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # building용 히스토리 (train 구간)
    hist = train_sub.copy()
    hist["일시"] = pd.to_datetime(hist["일시"])
    hist = hist.sort_values("일시").reset_index(drop=True)

    return model, proc, feature_cols, hist


# ===========================
# 4. 건물별 예측 (개별 + 그룹 앙상블)
# ===========================
def predict_for_building_ensemble(train_raw, test_raw, building_info,
                                  bld_id,
                                  building_model_pack,
                                  group_model_pack,
                                  w_building=0.6,
                                  w_group=0.4):

    train_sub = train_raw[train_raw["건물번호"] == bld_id].copy()
    test_sub = test_raw[test_raw["건물번호"] == bld_id].copy()

    if len(test_sub) == 0:
        return None

    # 개별 모델 언패킹
    if building_model_pack is not None:
        model_b, proc_b, feature_cols_b, hist_b = building_model_pack
    else:
        model_b = proc_b = feature_cols_b = hist_b = None

    # 그룹 모델 언패킹
    if group_model_pack is not None:
        model_g, proc_g, feature_cols_g = group_model_pack
        hist_g = train_sub.copy()
        hist_g["일시"] = pd.to_datetime(hist_g["일시"])
        hist_g = hist_g.sort_values("일시").reset_index(drop=True)
    else:
        model_g = proc_g = feature_cols_g = None
        hist_g = None

    # test 정렬
    test_sub = test_sub.copy()
    test_sub["일시"] = pd.to_datetime(test_sub["일시"])
    test_sub = test_sub.sort_values("일시").reset_index(drop=True)

    preds_list = []

    for i in range(len(test_sub)):
        cur = test_sub.iloc[[i]].copy()
        pred_b = None
        pred_g = None

        # ----------------- 개별 건물 모델 예측 -----------------
        if model_b is not None:
            cur_feat_b = proc_b.full_pipeline_test_step(cur, building_info)

            if len(hist_b) >= 168:
                cur_feat_b["lag_1"] = hist_b["전력소비량"].iloc[-1]
                cur_feat_b["lag_24"] = hist_b["전력소비량"].iloc[-24]
                cur_feat_b["lag_168"] = hist_b["전력소비량"].iloc[-168]
            else:
                last = hist_b["전력소비량"].iloc[-1] if len(hist_b) > 0 else 0.0
                cur_feat_b["lag_1"] = last
                cur_feat_b["lag_24"] = last
                cur_feat_b["lag_168"] = last

            X_test_b = cur_feat_b.drop(columns=["전력소비량", "일시", "num_date_time"],
                                       errors="ignore")
            X_test_b = X_test_b.select_dtypes(include=[np.number])
            for c in feature_cols_b:
                if c not in X_test_b.columns:
                    X_test_b[c] = 0
            X_test_b = X_test_b[feature_cols_b]

            pred_b = float(np.expm1(model_b.predict(X_test_b)[0]))

            new_row_b = cur.copy()
            new_row_b["전력소비량"] = pred_b
            hist_b = pd.concat([hist_b, new_row_b[hist_b.columns]], ignore_index=True)

        # ----------------- 사용그룹 공통 모델 예측 -----------------
        if model_g is not None:
            cur_feat_g = proc_g.full_pipeline_test_step(cur, building_info)

            if len(hist_g) >= 168:
                cur_feat_g["lag_1"] = hist_g["전력소비량"].iloc[-1]
                cur_feat_g["lag_24"] = hist_g["전력소비량"].iloc[-24]
                cur_feat_g["lag_168"] = hist_g["전력소비량"].iloc[-168]
            else:
                lastg = hist_g["전력소비량"].iloc[-1] if len(hist_g) > 0 else 0.0
                cur_feat_g["lag_1"] = lastg
                cur_feat_g["lag_24"] = lastg
                cur_feat_g["lag_168"] = lastg

            X_test_g = cur_feat_g.drop(columns=["전력소비량", "일시", "num_date_time"],
                                       errors="ignore")
            X_test_g = X_test_g.select_dtypes(include=[np.number])
            for c in feature_cols_g:
                if c not in X_test_g.columns:
                    X_test_g[c] = 0
            X_test_g = X_test_g[feature_cols_g]

            pred_g = float(np.expm1(model_g.predict(X_test_g)[0]))

            new_row_g = cur.copy()
            new_row_g["전력소비량"] = pred_g
            hist_g = pd.concat([hist_g, new_row_g[hist_g.columns]], ignore_index=True)

        # ----------------- 앙상블 -----------------
        if (pred_b is not None) and (pred_g is not None):
            pred = (w_building * pred_b + w_group * pred_g) / (w_building + w_group)
        elif pred_b is not None:
            pred = pred_b
        elif pred_g is not None:
            pred = pred_g
        else:
            pred = 0.0

        preds_list.append((cur["num_date_time"].iloc[0], pred))

    if not preds_list:
        return None

    return pd.DataFrame(preds_list, columns=["num_date_time", "answer"])


# ===========================
# 5. MAIN: 전체 실행
# ===========================
def main():
    train_path = "C:/data/train.csv"
    test_path = "C:/data/test.csv"
    building_path = "C:/data/building_info.csv"
    sample_path = "C:/data/sample_submission.csv"
    output_path = "C:/data/submission_xgb_group_building_ensemble.csv"

    train_raw = load_csv_with_fallback(train_path)
    test_raw = load_csv_with_fallback(test_path)
    building_info = load_csv_with_fallback(building_path)
    sample_sub = load_csv_with_fallback(sample_path)

    # 일조/일사 컬럼 정리
    train_raw = force_weather_columns(train_raw)
    test_raw = force_weather_columns(test_raw)
    building_info = force_weather_columns(building_info)

    # 타깃 컬럼 이름 통일
    if "전력소비량(kWh)" in train_raw.columns and "전력소비량" not in train_raw.columns:
        train_raw = train_raw.rename(columns={"전력소비량(kWh)": "전력소비량"})

    # 사용그룹 설정
    if "건물유형" not in building_info.columns:
        raise ValueError("building_info에 '건물유형' 컬럼이 필요합니다.")

    building_info["사용그룹"] = building_info["건물유형"].map(map_usage_group)

    # ----------------- 사용그룹별 모델 학습 -----------------
    group_models = {}
    for group_name in ["24시간", "주간", "기타"]:
        bld_ids = building_info.loc[
            building_info["사용그룹"] == group_name, "건물번호"
        ].unique()
        if len(bld_ids) == 0:
            continue

        print(f"\n=== 사용그룹 {group_name} (건물 수: {len(bld_ids)}) 모델 학습 ===")
        model_g, proc_g, feat_g = train_group_model_xgb(train_raw, building_info, bld_ids)
        if model_g is None:
            print(f"[{group_name}] 학습 데이터 부족 → 그룹 모델 스킵")
            continue
        group_models[group_name] = (model_g, proc_g, feat_g)

    # ----------------- 건물별 개별 모델 학습 -----------------
    building_models = {}
    print("\n=== 건물별 개별 모델 학습 시작 ===")
    for bld_id in sorted(train_raw["건물번호"].unique()):
        model_b, proc_b, feat_b, hist_b = train_building_model_xgb(
            train_raw, building_info, bld_id
        )
        if model_b is None:
            print(f"[건물 {bld_id}] 데이터 부족 → 개별 모델 스킵")
            continue
        building_models[bld_id] = (model_b, proc_b, feat_b, hist_b)
        print(f"[건물 {bld_id}] 개별 모델 학습 완료")

    # ----------------- 건물별 예측 (앙상블) -----------------
    all_preds = []
    print("\n▶ 건물별 예측 (개별 + 그룹 앙상블) 시작")

    for bld_id in sorted(test_raw["건물번호"].unique()):
        row_bi = building_info[building_info["건물번호"] == bld_id]
        if len(row_bi) == 0:
            group_pack = None
        else:
            group_name = row_bi["사용그룹"].iloc[0]
            group_pack = group_models.get(group_name, None)

        building_pack = building_models.get(bld_id, None)

        if (building_pack is None) and (group_pack is None):
            print(f"[건물 {bld_id}] 사용 가능한 모델 없음 → 스킵")
            continue

        pred_df = predict_for_building_ensemble(
            train_raw, test_raw, building_info,
            bld_id,
            building_pack,
            group_pack,
            w_building=0.6,
            w_group=0.4
        )
        if pred_df is not None:
            all_preds.append(pred_df)

    if not all_preds:
        raise RuntimeError("예측 결과가 비어 있습니다. 데이터/필터를 확인하세요.")

    all_preds_df = pd.concat(all_preds, ignore_index=True)

    submission = sample_sub.copy()
    submission = submission.drop(columns=["answer"], errors="ignore")
    submission = submission.merge(all_preds_df, on="num_date_time", how="left")

    if submission["answer"].isna().sum() > 0:
        submission["answer"].fillna(submission["answer"].mean(), inplace=True)

    submission.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ 제출 파일 생성 완료: {output_path}")


if __name__ == "__main__":
    main()
