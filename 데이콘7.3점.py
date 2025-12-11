import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
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
# 1-2. 건물별 타깃 클리핑
# ===========================
def clip_per_building(df, col="전력소비량", low_q=0.001, high_q=0.999):
    """
    건물번호별 전력소비량 분포 기준으로
    하위 low_q, 상위 high_q 바깥 값은 경계값으로 클리핑.
    시계열 행은 지우지 않고 값만 살짝 눌러줌.
    """
    def _clip(g):
        if col not in g.columns or g[col].isna().all():
            return g
        low = g[col].quantile(low_q)
        high = g[col].quantile(high_q)
        g[col] = g[col].clip(low, high)
        return g

    return df.groupby("건물번호", group_keys=False).apply(_clip)


# ===========================
# 2. 전처리 + 시계열 클래스 (공통)
# ===========================
class PowerPreprocessor:

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

        # 원본 컬럼명을 통일
        rename_map = {
            "기온(°C)": "기온",
            "습도(%)": "습도",
            "풍속(m/s)": "풍속",
            "강수량(mm)": "강수량",
            "일사(MJ/m2)": "일사"
        }
        for old, new in rename_map.items():
            if old in df.columns:
                df = df.rename(columns={old: new})

        df = df.replace("-", np.nan)

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

        # ---------------- 기본 파생 ----------------
        df["THI"] = (
            9/5 * df["기온"]
            - 0.55*(1 - df["습도"]/100)*(9/5*df["기온"] - 26)
            + 32
        )
        df["wind_chill"] = 13.12 + 0.6215*df["기온"] - 11.37*(df["풍속"]**0.16)
        df["cooling_ratio"] = df["냉방면적(m2)"] / (df["연면적(m2)"] + 1)
        df["solar_density"] = df["태양광용량(kW)"] / (df["연면적(m2)"] + 1)
        df["temp_sq"] = df["기온"] ** 2

        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        df["solar_effect"] = df["일사"] * df["태양광용량(kW)"]

        # ---------------- 냉방부하 지표 (Cooling Degree) ----------------
        base_temp = 24
        df["cooling_degree"] = np.maximum(df["기온"] - base_temp, 0)

        # ---------------- 업무시간 플래그(is_working_hour) ----------------
        def _is_working(row):
            btype = row.get("건물유형", "건물기타")
            if pd.isna(btype):
                btype = "건물기타"
            h = row["hour"]
            dow = row["dayofweek"]
            is_holiday = row["is_holiday"]

            if is_holiday == 1:
                return 0

            # 24시간 가동
            if btype in ["호텔", "병원", "통신시설", "전화국", "IDC(전화국)"]:
                return 1

            # 주간 (학교/연구소/백화점)
            if btype in ["학교", "연구소", "백화점"]:
                return int((dow < 5) and (8 <= h <= 18))

            # 공공: 9~18
            if btype in ["공공"]:
                return int((dow < 5) and (9 <= h <= 18))

            # 나머지(상용/아파트/기타)는 0
            return 0

        df["is_working_hour"] = df.apply(_is_working, axis=1)

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
        train = self.preprocess_datetime(train)
        train = self.merge_building_info(train, building_info)
        train = self.handle_missing_and_rename(train)
        train = self.create_features(train)
        train = self.add_lag_features(train)
        return train

    def full_pipeline_test_step(self, cur_row, building_info):
        df = cur_row.copy()
        df = self.preprocess_datetime(df)
        df = self.merge_building_info(df, building_info)
        df = self.handle_missing_and_rename(df)
        df = self.create_features(df)
        return df


# ===========================
# 3. 건물별 모델 학습 + val RMSE 비교
# ===========================
def train_building_models(train_raw, building_info, bld_id):
    """
    한 건물에 대해:
      - XGB, CatBoost 각각 학습
      - 8:2 시계열 split으로 val RMSE 비교
      - XGB / Cat / 평균 앙상블 중 최선 전략 선택
      - 선택 전략에 맞게 full-data로 최종 모델 재학습
    반환:
      best_strategy: "xgb" / "cat" / "ensemble"
      xgb_model (또는 None)
      cat_model (또는 None)
      feature_cols_xgb, feature_cols_cat
      hist (train 구간 원본 히스토리)
    """
    train_sub = train_raw[train_raw["건물번호"] == bld_id].copy()

    if len(train_sub) < 500:
        print(f"[건물 {bld_id}] 데이터 부족 → 스킵")
        return None

    proc = PowerPreprocessor()
    train_feat = proc.full_pipeline_train(train_sub, building_info)

    if "lag_168" in train_feat.columns:
        train_feat = train_feat.dropna(subset=["lag_168"]).reset_index(drop=True)

    if len(train_feat) < 300:
        print(f"[건물 {bld_id}] lag_168 이후 유효 데이터 부족 → 스킵")
        return None

    # 시계열 정렬
    if "일시" in train_feat.columns:
        train_feat = train_feat.sort_values("일시").reset_index(drop=True)

    drop_cols = [c for c in ["일시", "num_date_time"] if c in train_feat.columns]
    X_all = train_feat.drop(columns=["전력소비량"] + drop_cols, errors="ignore")
    X_all = X_all.select_dtypes(include=[np.number])
    y_all = train_feat["전력소비량"].astype(float)
    y_all_log = np.log1p(y_all)

    n = len(train_feat)
    split = int(n * 0.8)
    if split <= 0 or split >= n:
        print(f"[건물 {bld_id}] split 불가 → 스킵")
        return None

    X_tr, X_val = X_all.iloc[:split], X_all.iloc[split:]
    y_tr_log, y_val_log = y_all_log.iloc[:split], y_all_log.iloc[split:]

    feature_cols = X_all.columns.tolist()

    # ---------------- XGB 학습 + 검증 ----------------
    xgb_model = XGBRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=7,
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
    xgb_model.fit(X_tr, y_tr_log)
    pred_xgb_val_log = xgb_model.predict(X_val)
    y_val = np.expm1(y_val_log)
    y_pred_xgb = np.expm1(pred_xgb_val_log)
    rmse_xgb = np.sqrt(mean_squared_error(y_val, y_pred_xgb))

    # ---------------- CatBoost 학습 + 검증 ----------------
    cat_model = CatBoostRegressor(
        iterations=1200,
        learning_rate=0.04,
        depth=7,
        loss_function="RMSE",
        eval_metric="RMSE",
        l2_leaf_reg=5,
        random_seed=42,
        thread_count=-1,
        verbose=False
    )
    cat_model.fit(X_tr, y_tr_log)
    pred_cat_val_log = cat_model.predict(X_val)
    y_pred_cat = np.expm1(pred_cat_val_log)
    rmse_cat = np.sqrt(mean_squared_error(y_val, y_pred_cat))

    # ---------------- 단순 평균 앙상블 검증 ----------------
    y_pred_ens = 0.5 * y_pred_xgb + 0.5 * y_pred_cat
    rmse_ens = np.sqrt(mean_squared_error(y_val, y_pred_ens))

    print(f"[건물 {bld_id}] val RMSE - XGB: {rmse_xgb:.4f}, Cat: {rmse_cat:.4f}, Ensemble: {rmse_ens:.4f}")

    # ---------------- 최선 전략 선택 ----------------
    rmse_dict = {
        "xgb": rmse_xgb,
        "cat": rmse_cat,
        "ensemble": rmse_ens
    }
    best_strategy = min(rmse_dict, key=rmse_dict.get)
    print(f"[건물 {bld_id}] 선택 전략: {best_strategy}")

    # ---------------- full-data 재학습 ----------------
    xgb_final = None
    cat_final = None

    if best_strategy in ["xgb", "ensemble"]:
        xgb_final = XGBRegressor(
            n_estimators=1500,
            learning_rate=0.03,
            max_depth=7,
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
        xgb_final.fit(X_all, y_all_log)

    if best_strategy in ["cat", "ensemble"]:
        cat_final = CatBoostRegressor(
            iterations=1500,
            learning_rate=0.03,
            depth=8,
            loss_function="RMSE",
            eval_metric="RMSE",
            l2_leaf_reg=5,
            random_seed=42,
            thread_count=-1,
            verbose=False
        )
        cat_final.fit(X_all, y_all_log)

    # hist: 원본 train_sub 기준 (실제 전력소비량)
    hist = train_sub.copy()
    hist["일시"] = pd.to_datetime(hist["일시"])
    hist = hist.sort_values("일시").reset_index(drop=True)

    feature_cols_xgb = feature_cols
    feature_cols_cat = feature_cols

    return {
        "best_strategy": best_strategy,
        "xgb_model": xgb_final,
        "cat_model": cat_final,
        "feature_cols_xgb": feature_cols_xgb,
        "feature_cols_cat": feature_cols_cat,
        "hist": hist,
        "preprocessor": proc
    }


# ===========================
# 4. 건물별 예측 (선택 앙상블 전략 적용)
# ===========================
def predict_for_building_select_ensemble(train_raw, test_raw, building_info,
                                         bld_id,
                                         model_pack):

    test_sub = test_raw[test_raw["건물번호"] == bld_id].copy()
    if len(test_sub) == 0:
        return None

    best_strategy = model_pack["best_strategy"]
    xgb_model = model_pack["xgb_model"]
    cat_model = model_pack["cat_model"]
    feature_cols_xgb = model_pack["feature_cols_xgb"]
    feature_cols_cat = model_pack["feature_cols_cat"]
    hist = model_pack["hist"]
    proc = model_pack["preprocessor"]

    # test 정렬
    test_sub["일시"] = pd.to_datetime(test_sub["일시"])
    test_sub = test_sub.sort_values("일시").reset_index(drop=True)

    preds_list = []

    for i in range(len(test_sub)):
        cur = test_sub.iloc[[i]].copy()
        cur_feat = proc.full_pipeline_test_step(cur, building_info)

        # lag 생성: hist의 전력소비량 기준
        if "전력소비량" in hist.columns and len(hist) >= 1:
            last = hist["전력소비량"].iloc[-1]
        else:
            last = 0.0

        if "전력소비량" in hist.columns and len(hist) >= 24:
            last24 = hist["전력소비량"].iloc[-24]
        else:
            last24 = last

        if "전력소비량" in hist.columns and len(hist) >= 168:
            last168 = hist["전력소비량"].iloc[-168]
        else:
            last168 = last

        cur_feat["lag_1"] = last
        cur_feat["lag_24"] = last24
        cur_feat["lag_168"] = last168

        # 공통 기본 X (수치형만)
        X_test_all = cur_feat.drop(columns=["전력소비량", "일시", "num_date_time"],
                                   errors="ignore")
        X_test_all = X_test_all.select_dtypes(include=[np.number])

        pred_xgb = None
        pred_cat = None

        # XGB 예측
        if xgb_model is not None:
            X_b = X_test_all.copy()
            for c in feature_cols_xgb:
                if c not in X_b.columns:
                    X_b[c] = 0
            X_b = X_b[feature_cols_xgb]
            pred_log_b = xgb_model.predict(X_b)[0]
            pred_xgb = float(np.expm1(pred_log_b))

        # CatBoost 예측
        if cat_model is not None:
            X_c = X_test_all.copy()
            for c in feature_cols_cat:
                if c not in X_c.columns:
                    X_c[c] = 0
            X_c = X_c[feature_cols_cat]
            pred_log_c = cat_model.predict(X_c)[0]
            pred_cat = float(np.expm1(pred_log_c))

        # 선택 전략에 따른 최종 예측
        if best_strategy == "xgb":
            pred = pred_xgb
        elif best_strategy == "cat":
            pred = pred_cat
        else:  # ensemble
            if (pred_xgb is not None) and (pred_cat is not None):
                pred = 0.5 * pred_xgb + 0.5 * pred_cat
            elif pred_xgb is not None:
                pred = pred_xgb
            elif pred_cat is not None:
                pred = pred_cat
            else:
                pred = 0.0

        preds_list.append((cur["num_date_time"].iloc[0], pred))

        # hist 업데이트 (예측값으로 autoregressive 이어가기)
        new_row = cur.copy()
        new_row["전력소비량"] = pred
        common_cols = [c for c in hist.columns if c in new_row.columns]
        hist = pd.concat([hist, new_row[common_cols]], ignore_index=True)

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
    output_path = "C:/data/submission_building_select_xgb_cat_with_profile.csv"

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

    # 건물별 타깃 클리핑
    train_raw = clip_per_building(train_raw, col="전력소비량",
                                  low_q=0.001, high_q=0.999)

    # ================= 건물별 요일×시간대 평균 패턴 피처 추가 =================
    # train_raw 기준으로 건물별 dayofweek/hour 평균 부하 계산
    train_raw["일시"] = pd.to_datetime(train_raw["일시"])
    train_raw["hour"] = train_raw["일시"].dt.hour
    train_raw["dayofweek"] = train_raw["일시"].dt.dayofweek

    profile = (
        train_raw
        .groupby(["건물번호", "dayofweek", "hour"])["전력소비량"]
        .mean()
        .reset_index()
        .rename(columns={"전력소비량": "bld_dow_hour_mean"})
    )

    # test_raw에도 dayofweek/hour 생성 후 profile merge
    test_raw["일시"] = pd.to_datetime(test_raw["일시"])
    test_raw["hour"] = test_raw["일시"].dt.hour
    test_raw["dayofweek"] = test_raw["일시"].dt.dayofweek

    train_raw = train_raw.merge(
        profile,
        on=["건물번호", "dayofweek", "hour"],
        how="left"
    )
    test_raw = test_raw.merge(
        profile,
        on=["건물번호", "dayofweek", "hour"],
        how="left"
    )

    # 이후 PowerPreprocessor.preprocess_datetime에서 hour/dayofweek를 다시 계산하지만
    # bld_dow_hour_mean 컬럼은 그대로 유지되어 모델 입력으로 사용됨.

    # ================= 건물별 모델 학습 =================
    building_model_packs = {}
    print("\n=== 건물별 모델 학습 + 전략 선택 시작 ===")
    for bld_id in sorted(train_raw["건물번호"].unique()):
        pack = train_building_models(train_raw, building_info, bld_id)
        if pack is None:
            continue
        building_model_packs[bld_id] = pack

    # ================= 건물별 예측 =================
    print("\n=== 건물별 선택 앙상블 예측 시작 ===")
    all_preds = []
    for bld_id in sorted(test_raw["건물번호"].unique()):
        pack = building_model_packs.get(bld_id, None)
        if pack is None:
            print(f"[건물 {bld_id}] 학습된 모델 없음 → 스킵")
            continue
        pred_df = predict_for_building_select_ensemble(
            train_raw, test_raw, building_info, bld_id, pack
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
    print(f"\n✅ 건물별 XGB/Cat/Ensemble + 평균패턴 피처 제출 파일 생성 완료: {output_path}")


if __name__ == "__main__":
    main()


