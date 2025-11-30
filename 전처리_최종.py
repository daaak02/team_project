import os
import sys
import io
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ============================================================
# 한글 깨짐 방지 설정
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
# CSV 로딩 (여러 인코딩 시도)
# ============================================================
def load_csv_with_fallback(path):
    """
    여러 인코딩을 시도하여 CSV를 로드하는 함수.
    utf-8-sig, utf-8, cp949, euc-kr 순서로 시도.
    """
    encodings = ['utf-8-sig', 'utf-8', 'cp949', 'euc-kr']
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"Loaded '{path}' with encoding: {enc}")
            return df
        except Exception:
            continue

    # 마지막 시도: latin1
    df = pd.read_csv(path, encoding='latin1')
    print(f"Loaded '{path}' with fallback encoding: latin1")
    return df


# ============================================================
# 전처리 클래스
# ============================================================
class PowerDataPreprocessor:
    def __init__(self, building_info: pd.DataFrame):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

        # 건물유형 인코더는 building_info 기준으로 한 번만 fit
        if '건물유형' in building_info.columns:
            self.label_encoder.fit(building_info['건물유형'])
        else:
            self.label_encoder = None

    # -----------------------------
    # 1. 날짜/시간 특성 추출
    # -----------------------------
    def preprocess_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['일시'] = pd.to_datetime(df['일시'])

        df['year'] = df['일시'].dt.year
        df['month'] = df['일시'].dt.month
        df['day'] = df['일시'].dt.day
        df['hour'] = df['일시'].dt.hour
        df['dayofweek'] = df['일시'].dt.dayofweek
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

        # 계절 구분
        df['season'] = df['month'].apply(
            lambda x: 0 if x in [12, 1, 2] else
                      1 if x in [3, 4, 5] else
                      2 if x in [6, 7, 8] else
                      3
        )

        # 시간대 구분
        df['time_of_day'] = df['hour'].apply(
            lambda x: 0 if 6 <= x < 12 else   # 오전
                      1 if 12 <= x < 18 else  # 오후
                      2 if 18 <= x < 22 else  # 저녁
                      3                       # 심야
        )

        return df

    # -----------------------------
    # 2. 건물 정보 병합 + 인코딩
    # -----------------------------
    def merge_building_info(self, df: pd.DataFrame, building_info: pd.DataFrame) -> pd.DataFrame:
        df = df.merge(building_info, on='건물번호', how='left')

        if self.label_encoder is not None and '건물유형' in df.columns:
            # fit 은 이미 building_info에서 했고, 여기서는 transform만
            df['건물유형_encoded'] = self.label_encoder.transform(df['건물유형'].fillna(
                self.label_encoder.classes_[0]
            ))

        return df

    # -----------------------------
    # 3. 결측치 처리
    # -----------------------------
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # 기상 데이터: 건물번호 그룹 내에서 forward fill → backward fill
        weather_cols = ['기온', '강수량', '풍속', '습도', '일조', '일사']
        existing_weather_cols = [col for col in weather_cols if col in df.columns]

        for col in existing_weather_cols:
            df[col] = df.groupby('건물번호')[col].apply(
                lambda x: x.ffill().bfill()
            )

        # 건물 정보: '-', NaN → 숫자로 변환 후 중앙값, 필요시 0
        building_cols = ['연면적(m2)', '냉방면적(m2)', '태양광용량(kW)',
                         'ESS저장용량(kWh)', 'PCS용량(kW)']
        for col in building_cols:
            if col in df.columns:
                df[col] = df[col].replace('-', np.nan)
                df[col] = pd.to_numeric(df[col], errors='coerce')
                median = df[col].median()
                df[col] = df[col].fillna(median)
                df[col] = df[col].fillna(0)

        return df

    # -----------------------------
    # 4. 파생 특성 생성 (THI, wind_chill, 비율, 밀도)
    # -----------------------------
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # 불쾌지수 THI
        if {'기온', '습도'}.issubset(df.columns):
            T = df['기온']
            RH = df['습도']
            df['THI'] = 9/5 * T - 0.55 * (1 - RH / 100) * (9/5 * T - 26) + 32

        # 체감온도 wind_chill
        if {'기온', '풍속'}.issubset(df.columns):
            T = df['기온']
            V = df['풍속']
            df['wind_chill'] = (
                13.12 + 0.6215 * T
                - 11.37 * (V ** 0.16)
                + 0.3965 * T * (V ** 0.16)
            )

        # 냉방면적 비율
        if {'연면적(m2)', '냉방면적(m2)'}.issubset(df.columns):
            # 0 나누기 방지용 + 아주 작은 값
            df['냉방면적비율'] = df['냉방면적(m2)'] / (df['연면적(m2)'] + 1e-6)

        # 태양광 밀도
        if {'태양광용량(kW)', '연면적(m2)'}.issubset(df.columns):
            df['태양광밀도'] = df['태양광용량(kW)'] / (df['연면적(m2)'] + 1e-6)

        return df

    # -----------------------------
    # 5. 시간별/요일별 평균 전력소비량 (train 통계로 만들어 train/test 모두에 merge)
    # -----------------------------
    @staticmethod
    def add_time_means(train_df: pd.DataFrame,
                       test_df: pd.DataFrame,
                       target_col: str = '전력소비량'):
        """
        train의 건물번호-시간대/요일별 평균 전력소비량을 계산하여
        train, test 모두에 merge.
        """
        if target_col not in train_df.columns:
            print("[경고] 전력소비량 컬럼이 없어 시간별/요일별 평균 특성을 생성하지 않습니다.")
            return train_df, test_df

        # 건물번호 + hour 기준 평균
        hour_mean = (train_df
                     .groupby(['건물번호', 'hour'])[target_col]
                     .mean()
                     .reset_index()
                     .rename(columns={target_col: 'hour_mean'}))

        # 건물번호 + dayofweek 기준 평균
        dow_mean = (train_df
                    .groupby(['건물번호', 'dayofweek'])[target_col]
                    .mean()
                    .reset_index()
                    .rename(columns={target_col: 'dayofweek_mean'}))

        # train merge
        train_df = train_df.merge(hour_mean, on=['건물번호', 'hour'], how='left')
        train_df = train_df.merge(dow_mean, on=['건물번호', 'dayofweek'], how='left')

        # test merge (전력소비량은 없지만, train에서 만든 통계를 그대로 사용)
        test_df = test_df.merge(hour_mean, on=['건물번호', 'hour'], how='left')
        test_df = test_df.merge(dow_mean, on=['건물번호', 'dayofweek'], how='left')

        return train_df, test_df

    # -----------------------------
    # 6. 스케일링
    # -----------------------------
    def scale_features(self,
                       train_df: pd.DataFrame,
                       test_df: pd.DataFrame):
        train_df = train_df.copy()
        test_df = test_df.copy()

        numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()

        exclude_cols = [
            '건물번호', 'year', 'month', 'day', 'hour',
            'dayofweek', 'is_weekend', 'season', 'time_of_day',
            '건물유형_encoded'
        ]

        if '전력소비량' in numeric_cols:
            exclude_cols.append('전력소비량')

        scale_cols = [
            col for col in numeric_cols
            if col not in exclude_cols and col in test_df.columns
        ]

        if scale_cols:
            self.scaler.fit(train_df[scale_cols])
            train_df[scale_cols] = self.scaler.transform(train_df[scale_cols])
            test_df[scale_cols] = self.scaler.transform(test_df[scale_cols])

        return train_df, test_df


# ============================================================
# 이상치 제거 관련 함수 (IQR + 건물유형별 factor)
# ============================================================
def remove_outliers_iqr(df: pd.DataFrame, col: str, factor: float = 1.5):
    """IQR 기반 이상치 제거"""
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    mask = (df[col] < lower) | (df[col] > upper)
    print(f"  [IQR 제거] {col} 기준 이상치: {mask.sum()}건 제거")
    return df[~mask].reset_index(drop=True)


def remove_outliers_by_building_type(train: pd.DataFrame,
                                     target_col: str = '전력소비량'):
    """
    건물유형별로 IQR factor를 다르게 적용하여 이상치 제거.
    - 호텔, 병원, 백화점, IDC(전화국): factor=2.0  (느슨)
    - 학교, 연구소: factor=1.2  (엄격)
    - 그 외: factor=1.5 (표준)
    """
    print("\n" + "=" * 60)
    print("건물유형별 이상치 차등 제거 시작")
    print("=" * 60)

    df = train.copy()

    if '건물유형' not in df.columns:
        print("[경고] 건물유형 컬럼이 없어 일괄 IQR(1.5) 적용")
        return remove_outliers_iqr(df, target_col, factor=1.5)

    if target_col not in df.columns:
        print(f"[경고] {target_col} 컬럼이 없어 이상치 제거 불가")
        return df

    unique_types = df['건물유형'].unique()
    print(f"\n발견된 건물유형 ({len(unique_types)}개):")
    for btype in sorted(unique_types):
        print(f"  - {btype}: {len(df[df['건물유형'] == btype]):,}개")

    cleaned_list = []

    for btype in unique_types:
        sub = df[df['건물유형'] == btype].copy()
        original_count = len(sub)
        print(f"\n[{btype}] 처리 시작 (샘플 수: {original_count:,})")

        if btype in ['호텔', '병원', '백화점', 'IDC(전화국)']:
            print("  → 변동성 큼 → factor=2.0 적용")
            sub = remove_outliers_iqr(sub, target_col, factor=2.0)
        elif btype in ['학교', '연구소']:
            print("  → 변동성 작음 → factor=1.2 적용")
            sub = remove_outliers_iqr(sub, target_col, factor=1.2)
        else:
            print("  → 일반 건물 → factor=1.5 적용")
            sub = remove_outliers_iqr(sub, target_col, factor=1.5)

        removed = original_count - len(sub)
        rate = (removed / original_count * 100) if original_count > 0 else 0
        print(f"  제거율: {rate:.2f}% ({removed:,}/{original_count:,})")

        cleaned_list.append(sub)

    result = pd.concat(cleaned_list, axis=0).reset_index(drop=True)
    total_removed = len(df) - len(result)
    total_rate = (total_removed / len(df) * 100) if len(df) > 0 else 0

    print("\n" + "=" * 60)
    print("건물유형별 이상치 제거 완료")
    print("=" * 60)
    print(f"원본 데이터: {len(df):,}건")
    print(f"정제 데이터: {len(result):,}건")
    print(f"총 제거량: {total_removed:,}건 ({total_rate:.2f}%)")

    return result


# ============================================================
# 전체 파이프라인 함수
# ============================================================
def preprocess_loaded_data(train: pd.DataFrame,
                           test: pd.DataFrame,
                           building_info: pd.DataFrame):
    """
    이미 로드된 train/test/building_info에 대해
    - 날짜/시간 특성
    - 건물정보 병합 + 인코딩
    - 결측치 처리
    - 파생변수(THI, wind_chill, 냉방면적비율, 태양광밀도)
    - 이상치 제거 (train만)
    - 시간별/요일별 평균 특성 (train 통계 → train/test 모두)
    - 스케일링
    까지 수행.
    """
    print("=" * 60)
    print("전력 사용량 예측 데이터 전처리 시작")
    print("=" * 60)

    preprocessor = PowerDataPreprocessor(building_info)

    train_df = train.copy()
    test_df = test.copy()
    building_df = building_info.copy()

    print(f"\n[초기 데이터 크기]")
    print(f"Train: {train_df.shape}")
    print(f"Test: {test_df.shape}")
    print(f"Building info: {building_df.shape}")

    # 1. 날짜/시간 특성
    print("\n[1단계] 날짜/시간 특성 추출...")
    train_df = preprocessor.preprocess_datetime(train_df)
    test_df = preprocessor.preprocess_datetime(test_df)
    print("✓ 완료: year, month, day, hour, dayofweek, is_weekend, season, time_of_day")

    # 2. 건물 정보 병합
    print("\n[2단계] 건물 정보 병합...")
    train_df = preprocessor.merge_building_info(train_df, building_df)
    test_df = preprocessor.merge_building_info(test_df, building_df)
    print("✓ 완료: 건물유형, 연면적, 냉방면적, 태양광용량, ESS저장용량, PCS용량, 건물유형_encoded")

    # 3. 결측치 처리
    print("\n[3단계] 결측치 처리...")
    print(f"Train 결측치 수 (전): {train_df.isnull().sum().sum()}")
    print(f"Test 결측치 수 (전): {test_df.isnull().sum().sum()}")
    train_df = preprocessor.handle_missing_values(train_df)
    test_df = preprocessor.handle_missing_values(test_df)
    print(f"Train 결측치 수 (후): {train_df.isnull().sum().sum()}")
    print(f"Test 결측치 수 (후): {test_df.isnull().sum().sum()}")
    print("✓ 완료")

    # 4. 파생 특성 생성
    print("\n[4단계] 파생 특성 생성...")
    train_df = preprocessor.create_features(train_df)
    test_df = preprocessor.create_features(test_df)
    print("✓ 완료: THI, wind_chill, 냉방면적비율, 태양광밀도")

    # 5. 이상치 제거 (train만)
    print("\n[5단계] 이상치 제거 (train)...")
    if '전력소비량' in train_df.columns:
        before = len(train_df)
        train_df = remove_outliers_by_building_type(train_df, target_col='전력소비량')
        after = len(train_df)
        print(f"전력소비량 기준 이상치 제거: {before - after:,}건 제거 (잔여 {after:,}건)")
    else:
        print("전력소비량 컬럼 없음 → 이상치 제거 스킵")

    # 6. 시간별/요일별 평균 특성 (train 통계로 train/test 모두에 추가)
    print("\n[6단계] 시간별/요일별 평균 전력사용량 특성 추가...")
    train_df, test_df = PowerDataPreprocessor.add_time_means(train_df, test_df)
    print("✓ 완료: hour_mean, dayofweek_mean")

    # 7. 스케일링
    print("\n[7단계] 스케일링(StandardScaler) 적용...")
    train_processed, test_processed = preprocessor.scale_features(train_df, test_df)
    print("✓ 완료")

    # 최종 결과 요약
    print("\n" + "=" * 60)
    print("전처리 전체 완료")
    print("=" * 60)
    print(f"최종 Train shape: {train_processed.shape}")
    print(f"최종 Test  shape: {test_processed.shape}")
    print(f"\n최종 특성 수: {train_processed.shape[1]}")
    print("\n특성 목록:")
    for i, col in enumerate(train_processed.columns, 1):
        print(f"  {i}. {col}")

    return train_processed, test_processed


# ============================================================
# 실행 예시
# ============================================================
if __name__ == "__main__":
    # 경로는 각자 환경에 맞게 수정
    train_path = r'C:/Users/daaak/OneDrive/문서/바탕 화면/data/train.csv'
    test_path = r'C:/Users/daaak/OneDrive/문서/바탕 화면/data/test.csv'
    building_path = r'C:/Users/daaak/OneDrive/문서/바탕 화면/data/building_info.csv'

    train = load_csv_with_fallback(train_path)
    test = load_csv_with_fallback(test_path)
    building_info = load_csv_with_fallback(building_path)

    train_processed, test_processed = preprocess_loaded_data(train, test, building_info)

    # 필요시 저장
    train_processed.to_csv('train_processed.csv', index=False)
    test_processed.to_csv('test_processed.csv', index=False)
