# 02. 종속변수(Y) 만들기: 디지털 취약지수
# 입력: output/디지털취약계층_Y변수.xlsx (01번)
# 출력: output/Y변수_종합지수.csv
#
# 설문 응답으로 디지털 취약지수(0~100점)를 계산하고, 점수로 3집단(취약/경계선/비취약)을 나눔
# 지수 계산 방식: 한국지능정보사회진흥원, 「2025 디지털정보격차 실태조사 보고서」

import os
import pandas as pd
from skimage.filters import threshold_multiotsu   # pip install scikit-image

y = pd.read_excel('output/디지털취약계층_Y변수.xlsx')


def cols(prefix, n):
    """cols('Q6_', 3) → ['Q6_1', 'Q6_2', 'Q6_3']"""
    return [f'{prefix}{i}' for i in range(1, n + 1)]


def score_1to4(columns):
    """1~4점 문항 여러 개의 평균을 0~100점으로 바꾸기 (1점→0, 2점→33.3, 3점→66.7, 4점→100)"""
    return (y[columns].mean(axis=1) - 1) / 3 * 100


# 1. 접근지수 = 인터넷 상시 접속(50%) + 기기 보유(50%)
접속 = (y['Q3'] == 1) * 100                                   # 집에서 인터넷 가능하면 100점
기기 = y[['Q1_1', 'Q1_2', 'Q2K1_1_1', 'Q2K1_1_2']].mean(axis=1) * 100
#      데스크탑, 노트북, 스마트폰, 피처폰 중 가진 비율 (예: 4개 중 2개 → 50점)
y['접근지수'] = 0.5 * 접속 + 0.5 * 기기

# 2. 역량지수 = PC 이용능력(50%) + 모바일 이용능력(50%)
pc = score_1to4(cols('Q6_', 7))          # 소프트웨어 설치, 파일 전송 등 7문항
mobile = score_1to4(cols('Q7_', 7))      # 와이파이 설정, 앱 설치 등 7문항
y['역량지수'] = 0.5 * pc + 0.5 * mobile

# 3. 활용지수 = 인터넷 이용 여부(40%) + 서비스 다양성(40%) + 심화 활용(20%)
이용 = y['Q10'] * 100                                          # 이용하면 100점, 안 하면 0점

다양성 = score_1to4([c for c in y.columns if c[:3] in ['Q12', 'Q13', 'Q14']])    # 검색·SNS·쇼핑·금융 등
심화 = score_1to4([c for c in y.columns if c[:3] in ['Q15', 'Q16', 'Q17', 'Q18']])   # 콘텐츠 제작·사회참여·경제활동 등

y['활용지수'] = 0.4 * 이용 + 0.4 * 다양성 + 0.2 * 심화

# 4. 종합지수 = 접근(20%) + 역량(40%) + 활용(40%)   → 0~100점, 낮을수록 취약
y['종합지수'] = 0.2 * y['접근지수'] + 0.4 * y['역량지수'] + 0.4 * y['활용지수']

# 5. 3집단으로 나누기 (Multi-Otsu)
#    점수 분포를 보고 '집단 안은 비슷하게, 집단끼리는 최대한 다르게' 나뉘는 기준점 2개를 찾음
t1, t2 = threshold_multiotsu(y['종합지수'].values, classes=3)
print(f'기준점: {t1:.2f}점, {t2:.2f}점')

y['등급'] = 2                              # 비취약
y.loc[y['종합지수'] <= t2, '등급'] = 1      # 경계선
y.loc[y['종합지수'] <= t1, '등급'] = 0      # 취약

print(y['등급'].value_counts(normalize=True).sort_index().round(3))

# 6. 저장 (등급은 모델링 단계에서 같은 방법으로 다시 나눔)
os.makedirs('output', exist_ok=True)
y[['응답ID', 'TYP', '접근지수', '역량지수', '활용지수', '종합지수']].to_csv(
    'output/Y변수_종합지수.csv', index=False, encoding='utf-8-sig')
