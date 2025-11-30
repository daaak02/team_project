import pandas as pd
import numpy as np

# ============================================================
# IQR 기반 이상치 제거 함수
# ============================================================
def remove_outliers_iqr(df, col, factor=1.5):
    """IQR 기반 이상치 제거"""
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    mask = (df[col] < lower) | (df[col] > upper)
    print(f"  [IQR 제거] {col} 기준 이상치: {mask.sum()}건 제거")
    return df[~mask].reset_index(drop=True)


# ============================================================
# Z-score 기반 이상치 제거 함수
# ============================================================
def remove_outliers_zscore(df, col, threshold=3.0):
    """Z-Score 기반 이상치 제거"""
    mean = df[col].mean()
    std = df[col].std()
    if std == 0:
        print(f"  [Z-score 제거] {col} 표준편차 0 → 스킵")
        return df
    z = (df[col] - mean) / std
    mask = np.abs(z) > threshold
    print(f"  [Z-score 제거] {col} 기준 이상치: {mask.sum()}건 제거")
    return df[~mask].reset_index(drop=True)


# ============================================================
# 건물유형별 이상치 규칙 차등 적용 (실제 데이터 기반)
# ============================================================
def remove_outliers_by_building_type(train, target_col='전력소비량'):
    """
    실제 건물 유형에 맞춘 이상치 제거
    
    건물 유형 분류:
    - 변동성 큰 건물: 호텔, 병원, 백화점, IDC
    - 변동성 작은 건물: 학교, 연구소
    - 일반 건물: 상용, 아파트, 공공, 건축커뮤니티, 인프라
    """
    print("\n" + "="*60)
    print("건물유형별 이상치 차등 제거 시작")
    print("="*60)

    df = train.copy()
    
    # 컬럼 확인
    if "건물유형" not in df.columns:
        print("[경고] 건물유형 컬럼이 없음 → 일괄 IQR 적용")
        return remove_outliers_iqr(df, target_col, factor=1.5)
    
    if target_col not in df.columns:
        print(f"[경고] {target_col} 컬럼이 없음 → 이상치 제거 불가")
        return df

    # 건물 유형별로 분류
    unique_types = df["건물유형"].unique()
    print(f"\n발견된 건물 유형 ({len(unique_types)}개):")
    for btype in sorted(unique_types):
        count = len(df[df["건물유형"] == btype])
        print(f"  - {btype}: {count:,}개")
    
    cleaned_list = []

    for btype in unique_types:
        sub = df[df["건물유형"] == btype].copy()
        original_count = len(sub)
        print(f"\n[{btype}] 처리 시작 (샘플 수: {original_count:,})")

        # 변동성이 큰 건물 (24시간 운영, 특수 이벤트 많음)
        if btype in ["호텔", "병원", "백화점", "IDC(전화국)"]:
            print(f"  → 변동성 큼 → factor=2.0 (느슨한 기준)")
            sub = remove_outliers_iqr(sub, target_col, factor=2.0)
        
        # 변동성이 작은 건물 (규칙적 패턴)
        elif btype in ["학교", "연구소"]:
            print(f"  → 변동성 작음 → factor=1.2 (엄격한 기준)")
            sub = remove_outliers_iqr(sub, target_col, factor=1.2)
        
        # 일반 건물 (표준 기준)
        else:
            print(f"  → 일반 건물 → factor=1.5 (표준 기준)")
            sub = remove_outliers_iqr(sub, target_col, factor=1.5)
        
        removed = original_count - len(sub)
        removal_rate = (removed / original_count * 100) if original_count > 0 else 0
        print(f"  제거율: {removal_rate:.2f}% ({removed:,}/{original_count:,})")
        
        cleaned_list.append(sub)

    result = pd.concat(cleaned_list, axis=0).reset_index(drop=True)
    
    total_removed = len(df) - len(result)
    total_removal_rate = (total_removed / len(df) * 100) if len(df) > 0 else 0
    
    print("\n" + "="*60)
    print("건물유형별 이상치 제거 완료")
    print("="*60)
    print(f"원본 데이터: {len(df):,}건")
    print(f"정제 데이터: {len(result):,}건")
    print(f"총 제거량: {total_removed:,}건 ({total_removal_rate:.2f}%)")

    return result


# ============================================================
# 건물유형별 통계 분석
# ============================================================
def analyze_building_types(train, target_col='전력소비량'):
    """건물 유형별 전력 사용 패턴 분석"""
    
    if '건물유형' not in train.columns or target_col not in train.columns:
        print("필요한 컬럼이 없습니다.")
        return
    
    print("\n" + "="*60)
    print("건물유형별 전력 사용량 통계")
    print("="*60)
    
    stats = train.groupby('건물유형')[target_col].agg([
        ('개수', 'count'),
        ('평균', 'mean'),
        ('중앙값', 'median'),
        ('표준편차', 'std'),
        ('최소', 'min'),
        ('최대', 'max'),
        ('변동계수(CV)', lambda x: x.std() / x.mean() if x.mean() != 0 else 0)
    ]).round(2)
    
    # 변동계수로 정렬 (변동성 큰 순서)
    stats = stats.sort_values('변동계수(CV)', ascending=False)
    
    print(stats)
    
    print("\n💡 변동계수(CV) 해석:")
    print("  - CV > 0.5: 변동성 매우 큼 → 느슨한 기준(2.0) 추천")
    print("  - 0.3 < CV < 0.5: 변동성 보통 → 표준 기준(1.5)")
    print("  - CV < 0.3: 변동성 작음 → 엄격한 기준(1.2) 추천")
    
    return stats


# ============================================================
# 사용 예시
# ============================================================
if __name__ == "__main__":
    
    print("="*60)
    print("이상치 제거 코드 (실제 데이터 기반)")
    print("="*60)
    
    # 사용 방법
    print("\n[사용법 1] 건물유형별 차등 적용 (추천)")
    print("-"*60)
    print("train_cleaned = remove_outliers_by_building_type(train)")
    
    print("\n[사용법 2] 먼저 통계 분석 후 적용")
    print("-"*60)
    print("stats = analyze_building_types(train)")
    print("train_cleaned = remove_outliers_by_building_type(train)")
    
    print("\n[사용법 3] 일괄 IQR 적용")
    print("-"*60)
    print("train_cleaned = remove_outliers_iqr(train, '전력소비량', factor=1.5)")
    
    print("\n[사용법 4] Z-Score 적용")
    print("-"*60)
    print("train_cleaned = remove_outliers_zscore(train, '전력소비량', threshold=3.0)")