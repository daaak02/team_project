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

#데이터 정보 출력
print(train.shape, test.shape, building_info.shape)
print(train.head(), test.head(), building_info.head())

#전처리
class PowerDataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
    
    def preprocess_datetime(self, df):
        """날짜/시간 특성 추출"""
        df = df.copy()
        df['일시'] = pd.to_datetime(df['일시'])
        
        # 시간 관련 특성
        df['year'] = df['일시'].dt.year
        df['month'] = df['일시'].dt.month
        df['day'] = df['일시'].dt.day
        df['hour'] = df['일시'].dt.hour
        df['dayofweek'] = df['일시'].dt.dayofweek
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        
        # 계절 구분
        df['season'] = df['month'].apply(lambda x: 
            0 if x in [12, 1, 2] else  # 겨울
            1 if x in [3, 4, 5] else   # 봄
            2 if x in [6, 7, 8] else   # 여름
            3)                          # 가을
        
        # 시간대 구분
        df['time_of_day'] = df['hour'].apply(lambda x:
            0 if 6 <= x < 12 else   # 오전
            1 if 12 <= x < 18 else  # 오후
            2 if 18 <= x < 22 else  # 저녁
            3)                       # 심야
        
        return df
    
    def merge_building_info(self, df, building_info):
        """건물 정보 병합"""
        df = df.merge(building_info, on='건물번호', how='left')
        
        # 건물유형 인코딩
        if '건물유형' in df.columns:
            df['건물유형_encoded'] = self.label_encoder.fit_transform(df['건물유형'])
        
        return df
    
    def handle_missing_values(self, df):
        """결측치 처리"""
        df = df.copy()
        
        # 기상 데이터 결측치를 forward fill 후 backward fill
        weather_cols = ['기온', '강수량', '풍속', '습도', '일조', '일사']
        existing_weather_cols = [col for col in weather_cols if col in df.columns]
        
        for col in existing_weather_cols:
            df[col] = df.groupby('건물번호')[col].fillna(method='ffill').fillna(method='bfill')
        
        # 건물 정보 결측치는 0 또는 중앙값으로 대체
        building_cols = ['연면적(m2)', '냉방면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)']
        for col in building_cols:
            if col in df.columns:
                # '-' 문자를 NaN으로 변환
                df[col] = df[col].replace('-', np.nan)
                # 숫자로 변환
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # 중앙값으로 결측치 대체
                df[col] = df[col].fillna(df[col].median())
                # 중앙값도 NaN이면 0으로 대체
                df[col] = df[col].fillna(0)
        
        return df
    
    def create_features(self, df):
        """추가 특성 생성"""
        df = df.copy()
        
        # 불쾌지수 (Temperature-Humidity Index)
        if '기온' in df.columns and '습도' in df.columns:
            df['THI'] = 9/5 * df['기온'] - 0.55 * (1 - df['습도']/100) * (9/5 * df['기온'] - 26) + 32
        
        # 체감온도 (Wind Chill)
        if '기온' in df.columns and '풍속' in df.columns:
            df['wind_chill'] = 13.12 + 0.6215 * df['기온'] - 11.37 * (df['풍속'] ** 0.16) + 0.3965 * df['기온'] * (df['풍속'] ** 0.16)
        
        # 건물 효율 지표
        if '연면적(m2)' in df.columns and '냉방면적(m2)' in df.columns:
            df['냉방면적비율'] = df['냉방면적(m2)'] / (df['연면적(m2)'] + 1)
        
        if '태양광용량(kW)' in df.columns and '연면적(m2)' in df.columns:
            df['태양광밀도'] = df['태양광용량(kW)'] / (df['연면적(m2)'] + 1)
        
        # 시간별 평균 전력소비량 (train에만 존재)
        if '전력소비량' in df.columns:
            df['hour_mean'] = df.groupby(['건물번호', 'hour'])['전력소비량'].transform('mean')
            df['dayofweek_mean'] = df.groupby(['건물번호', 'dayofweek'])['전력소비량'].transform('mean')
        
        return df
    
    def create_lag_features(self, df, target_col='전력소비량'):
        """시계열 lag 특성 생성 (train 데이터에만 적용)"""
        if target_col not in df.columns:
            return df
        
        df = df.copy()
        df = df.sort_values(['건물번호', '일시'])
        
        # 1시간, 24시간, 168시간(1주일) 전 데이터
        for lag in [1, 24, 168]:
            df[f'lag_{lag}'] = df.groupby('건물번호')[target_col].shift(lag)
        
        # 이동 평균
        for window in [24, 168]:
            df[f'rolling_mean_{window}'] = df.groupby('건물번호')[target_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
        
        return df
    
    def scale_features(self, train_df, test_df):
        """특성 스케일링"""
        train_df = train_df.copy()
        test_df = test_df.copy()
        
        # 스케일링할 컬럼 선택
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # 제외할 컬럼
        exclude_cols = ['건물번호', 'year', 'month', 'day', 'hour', 'dayofweek', 
                       'is_weekend', 'season', 'time_of_day', '건물유형_encoded']
        
        if '전력소비량' in numeric_cols:
            exclude_cols.append('전력소비량')
        
        scale_cols = [col for col in numeric_cols if col not in exclude_cols and col in test_df.columns]
        
        if scale_cols:
            train_df[scale_cols] = self.scaler.fit_transform(train_df[scale_cols])
            test_df[scale_cols] = self.scaler.transform(test_df[scale_cols])
        
        return train_df, test_df


def preprocess_loaded_data(train, test, building_info, use_lag=False):
    """이미 로드된 데이터를 전처리하는 함수
    
    Parameters:
    -----------
    train : DataFrame
        학습 데이터
    test : DataFrame
        테스트 데이터
    building_info : DataFrame
        건물 정보 데이터
    use_lag : bool, default=False
        시계열 lag 특성 사용 여부
    
    Returns:
    --------
    train_processed, test_processed : DataFrames
        전처리된 학습 및 테스트 데이터
    """
    
    print("=" * 60)
    print("전력 사용량 예측 데이터 전처리 시작")
    print("=" * 60)
    
    preprocessor = PowerDataPreprocessor()
    
    # 데이터 복사 (원본 보존)
    train_df = train.copy()
    test_df = test.copy()
    building_df = building_info.copy()
    
    print(f"\n[초기 데이터 크기]")
    print(f"Train: {train_df.shape}")
    print(f"Test: {test_df.shape}")
    print(f"Building info: {building_df.shape}")
    
    # 1. 날짜/시간 특성 추출
    print("\n[1단계] 날짜/시간 특성 추출 중...")
    train_df = preprocessor.preprocess_datetime(train_df)
    test_df = preprocessor.preprocess_datetime(test_df)
    print("✓ 완료: year, month, day, hour, dayofweek, is_weekend, season, time_of_day")
    
    # 2. 건물 정보 병합
    print("\n[2단계] 건물 정보 병합 중...")
    train_df = preprocessor.merge_building_info(train_df, building_df)
    test_df = preprocessor.merge_building_info(test_df, building_df)
    print("✓ 완료: 건물유형, 연면적, 냉방면적, 태양광용량, ESS저장용량, PCS용량")
    
    # 3. 결측치 처리
    print("\n[3단계] 결측치 처리 중...")
    print(f"Train 결측치 (처리 전): {train_df.isnull().sum().sum()}")
    print(f"Test 결측치 (처리 전): {test_df.isnull().sum().sum()}")
    train_df = preprocessor.handle_missing_values(train_df)
    test_df = preprocessor.handle_missing_values(test_df)
    print(f"Train 결측치 (처리 후): {train_df.isnull().sum().sum()}")
    print(f"Test 결측치 (처리 후): {test_df.isnull().sum().sum()}")
    print("✓ 완료")
    
    # 4. 추가 특성 생성
    print("\n[4단계] 추가 특성 생성 중...")
    train_df = preprocessor.create_features(train_df)
    test_df = preprocessor.create_features(test_df)
    print("✓ 완료: THI, wind_chill, 냉방면적비율, 태양광밀도")
    
    # 5. Lag 특성 (선택적)
    if use_lag:
        print("\n[5단계] Lag 특성 생성 중...")
        train_df = preprocessor.create_lag_features(train_df)
        print("✓ 완료: lag_1, lag_24, lag_168, rolling_mean_24, rolling_mean_168")
        print("⚠ 주의: Lag 특성으로 인해 초기 데이터에 결측치가 생성됩니다.")
    
    # 6. 스케일링
    print("\n[6단계] 특성 스케일링 중...")
    train_processed, test_processed = preprocessor.scale_features(train_df, test_df)
    print("✓ 완료: StandardScaler 적용")
    
    # 최종 결과
    print("\n" + "=" * 60)
    print("전처리 완료!")
    print("=" * 60)
    print(f"Train shape: {train_processed.shape}")
    print(f"Test shape: {test_processed.shape}")
    print(f"\n생성된 총 특성 수: {train_processed.shape[1]}")
    print(f"\n특성 목록:")
    for i, col in enumerate(train_processed.columns, 1):
        print(f"  {i}. {col}")
    
    return train_processed, test_processed


# ============================================================
# 사용 예시
# ============================================================

if __name__ == "__main__":
    #데이터 불러오기
    train = load_csv_with_fallback(r'C:/Users/daaak/OneDrive/문서/바탕 화면/data/train.csv')
    test = load_csv_with_fallback(r'C:/Users/daaak/OneDrive/문서/바탕 화면/data/test.csv')
    building_info = load_csv_with_fallback(r'C:/Users/daaak/OneDrive/문서/바탕 화면/data/building_info.csv')
    
    #전처리 실행
    train_processed, test_processed = preprocess_loaded_data(
        train, 
        test, 
        building_info,
        use_lag=False  # Lag 특성 사용 여부
    )
    
    #처리된 데이터 저장 (선택사항)
    train_processed.to_csv('train_processed.csv', index=False)
    test_processed.to_csv('test_processed.csv', index=False)
    
    
# 1. 데이터 형태 확인
print("=" * 60)
print("전처리 완료 확인")
print("=" * 60)
print(f"Train shape: {train_processed.shape}")
print(f"Test shape: {test_processed.shape}")

# 2. 상위 데이터 확인
print("\nTrain 데이터 샘플:")
print(train_processed.head())

print("\nTest 데이터 샘플:")
print(test_processed.head())

# 3. 기본 통계
print("\n기본 통계:")
print(train_processed.describe())

