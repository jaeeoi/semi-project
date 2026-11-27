# 01. 데이터 전처리
# 입력: data/ 폴더의 원본 엑셀 6개 (2025 디지털정보격차 실태조사)
# 출력: output/디지털취약계층_X변수.xlsx, output/디지털취약계층_Y변수.xlsx

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


# 1. 원본 불러오기
path = 'data/2025년 디지털정보격차 실태조사_'
gen = pd.read_excel(path + '1) 일반국민_DATA.xlsx')
farm = pd.read_excel(path + '2) 농어민_DATA.xlsx')
dis = pd.read_excel(path + '3) 장애인_DATA.xlsx')
low = pd.read_excel(path + '4) 저소득층_DATA.xlsx')
nk = pd.read_excel(path + '5) 북한이탈주민_DATA.xlsx')
mar = pd.read_excel(path + '6) 결혼이민자_DATA.xlsx')
# 고령층 파일은 일반국민 안에 이미 포함돼 있어서(응답ID 중복) 사용 안 함


# 2. 인구통계 문항 이름 통일
#    계층마다 같은 질문의 문항 번호가 달라서 이름을 맞춤
#    탈북민·이민자는 연령·성별만 조사됨
#    '가구형태', '가구원수_응답'은 5-7에서 '가구원수'로 합치기 전 임시 이름
gen = gen.rename(columns={'ADQ1': '연령', 'ADQ2': '성별', 'ADQ3': '직업', 'ADQ4': '학력',
                          'ADQ9': '가구소득', 'ADQ101': '거주지역', 'ADQ103': '지역구분_읍면동',
                          'ADQ8_1': '가구형태', 'ADQ8_1T': '가구원수_응답'})
farm = farm.rename(columns={'BDQ2': '연령', 'BDQ3': '성별', 'BDQ4': '학력',
                            'BDQ9': '가구소득', 'BDQ101': '거주지역', 'BDQ103': '지역구분_읍면동',
                            'BDQ8': '가구형태', 'BDQ8T': '가구원수_응답'})
dis = dis.rename(columns={'CDQ1': '연령', 'CDQ2': '성별', 'CDQ6': '직업', 'CDQ7': '학력',
                          'CDQ15': '가구소득', 'CDQ161': '거주지역', 'CDQ163': '지역구분_읍면동',
                          'CDQ13': '가구형태', 'CDQ13T': '가구원수_응답'})
low = low.rename(columns={'DDQ1': '연령', 'DDQ2': '성별', 'DDQ3': '직업', 'DDQ4': '학력',
                          'DDQ9': '가구소득', 'DDQ101': '거주지역', 'DDQ103': '지역구분_읍면동',
                          'DDQ8': '가구형태', 'DDQ8T': '가구원수_응답'})
nk = nk.rename(columns={'EDQ1': '연령', 'EDQ2': '성별'})
mar = mar.rename(columns={'FDQ1': '연령', 'FDQ2': '성별'})

# 원본 문항에 없는 값 채우기
farm['직업'] = 6          # 농어민: 직업 문항 없음 → 전원 농림어업(6)
dis['장애여부'] = 1        # 장애인 계층: 전원 1
for d in [farm, low, nk, mar]:
    d['장애여부'] = 0      # 나머지 계층: 전원 0


# 3. 하나로 합치기 + 빈칸 정리
df = pd.concat([gen, farm, dis, low, nk, mar], ignore_index=True)

# 원본의 빈칸은 공백(' ')으로 들어 있음 → 결측(NaN)으로 바꿈
df = df.replace(r'^\s*$', np.nan, regex=True).infer_objects()
print('합친 크기:', df.shape)


# 4. 행 제거
# 일반국민 장애여부: ADQ5 1=없음 → 0, 2=있음 → 1
df.loc[df['TYP'] == 1, '장애여부'] = (df['ADQ5'] == 2).astype(int)

# 일반국민 중 '장애 있음' 제거 (장애인 계층과 겹침)
df = df[~((df['TYP'] == 1) & (df['장애여부'] == 1))].copy()
print('행 제거 후:', df.shape)


# 5. 값 바꾸기
# 5-1. 예/아니오(1/2) → 1/0
for c in ['Q1_1', 'Q1_2', 'Q25']:
    df[c] = (df[c] == 1).astype(int)

# 5-2. 여러 개 고르는 문항 → 골랐으면 1, 안 골랐으면 0
multi = ['Q2K1_1_1', 'Q2K1_1_2', 'Q4B_1_1', 'Q4B_2_1',
         'Q5K1_1', 'Q5K1_2', 'Q5K1_3', 'Q5K1_4', 'Q5K1_5', 'Q5K1_9997',
         'Q27A_1_1', 'Q27A_2_1', 'Q27A_3_1', 'Q27A_4_1',
         'Q27A_5_1', 'Q27A_6_1', 'Q27A_7_1', 'Q27A_8_1',
         'Q28_1', 'Q28_2', 'Q28_3', 'Q28_4', 'Q28_5', 'Q28_6', 'Q28_7', 'Q28_9997',
         'Q31_1', 'Q31_2', 'Q31_3', 'Q31_4']
df[multi] = df[multi].notna().astype(int)
df = df.rename(columns={'Q5K1_9997': 'Q5K1_6'})   # 기타 → Q5K1_6

# 5-3. 최근 인터넷 이용(Q10): 1·2(이용함) → 1, 3(이용 안 함) → 0
df['Q10'] = df['Q10'].isin([1, 2]).astype(int)

# 5-4. 서비스 이용 정도(Q12~Q18) 빈칸 → 1(전혀 이용 안 함)
#      빈칸 = 최근 한 달 인터넷을 안 쓴 사람
service = [c for c in df.columns if c[:3] in ['Q12', 'Q13', 'Q14', 'Q15', 'Q16', 'Q17', 'Q18']]
df[service] = df[service].fillna(1)

# 5-5. AI 사용 빈도(Q27B) 빈칸 → 0 (그 AI를 몰라서 응답 안 함)
q27b = ['Q27B_1', 'Q27B_2', 'Q27B_3', 'Q27B_4', 'Q27B_5', 'Q27B_6', 'Q27B_7', 'Q27B_8']
df[q27b] = df[q27b].fillna(0)

# 5-6. 직업 '기타(9997)' → 15
df['직업'] = df['직업'].replace(9997, 15)

# 5-7. 가구원수: 1인가구면 1, 다인가구면 응답한 숫자 (다인가구인데 빈칸이면 2)
df['가구원수'] = np.where(df['가구형태'] == 1, 1, df['가구원수_응답'].fillna(2))
df.loc[df['TYP'].isin([5, 6]), '가구원수'] = np.nan   # 탈북민·이민자는 문항 없음

# 그 외 빈칸(요금 부담 Q4C, 학력, 소득 등)은 그대로 둠


# 6. X변수 / Y변수 나눠서 저장
#    제외한 문항: Y와 비슷한 문항(Q2K2, Q19), 데이터 누수(Q8, Q9),
#    취약성의 결과에 가까움(Q23, Q24), 목적과 무관(ADQ3A, ADQ6~8 등)
def cols(prefix, n):
    """cols('Q5_', 3) → ['Q5_1', 'Q5_2', 'Q5_3']"""
    return [f'{prefix}{i}' for i in range(1, n + 1)]


x_cols = (['응답ID', 'TYP', 'Q4B_1_1', 'Q4B_2_1', 'Q4C_1', 'Q4C_2']
          + cols('Q5_', 5) + cols('Q5K1_', 6) + cols('Q11_', 3)
          + cols('Q20_', 10) + cols('Q21_', 4) + cols('Q22_', 4)
          + ['Q25'] + cols('Q26_', 12)
          + [f'Q27A_{i}_1' for i in range(1, 9)] + cols('Q27B_', 8) + cols('Q27C_', 8)
          + cols('Q28_', 7) + ['Q28_9997'] + cols('Q29_', 10) + ['Q30'] + cols('Q31_', 4)
          + ['연령', '성별', '직업', '학력', '장애여부', '가구원수',
             '가구소득', '거주지역', '지역구분_읍면동', 'WT'])
# Q11, Q26, WT는 모델링 단계에서 제거

y_cols = (['응답ID', 'TYP', 'Q1_1', 'Q1_2', 'Q2K1_1_1', 'Q2K1_1_2', 'Q3']
          + cols('Q6_', 7) + cols('Q7_', 7) + ['Q10'] + service)

os.makedirs('output', exist_ok=True)
df[x_cols].to_excel('output/디지털취약계층_X변수.xlsx', index=False)
df[y_cols].to_excel('output/디지털취약계층_Y변수.xlsx', index=False)
print('X변수:', df[x_cols].shape)
print('Y변수:', df[y_cols].shape)
