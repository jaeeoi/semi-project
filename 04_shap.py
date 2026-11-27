# 04. SHAP 분석
# 입력: output/디지털취약계층_X변수.xlsx (01번), output/Y변수_종합지수.csv (02번)
# 출력: output/shap_bar_top35.png, output/shap_beeswarm_취약.png
#
# 최종 LightGBM 모델이 사람을 '취약' / '경계선'으로 판단할 때, 변수 하나하나가 얼마나 기여했는지 확인

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from skimage.filters import threshold_multiotsu
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier

warnings.filterwarnings('ignore')

# 한글 폰트 (Windows: 'Malgun Gothic' / Mac: 'AppleGothic')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


# 0. 데이터 준비 (03번과 같음)
# 0-1. X변수 파일 + 종합지수 파일 합치기 (응답ID 기준)
df_x = pd.read_excel('output/디지털취약계층_X변수.xlsx')
df_y = pd.read_csv('output/Y변수_종합지수.csv')
df = df_x.merge(df_y[['응답ID', '종합지수']], on='응답ID')

# 0-2. 공백(' ')이 있으면 결측으로 바꾸고 전부 숫자로
df = df.replace(' ', np.nan).apply(pd.to_numeric, errors='coerce')

# 0-3. 정답(y): 종합지수를 02번과 같은 기준점으로 3집단
t1, t2 = threshold_multiotsu(df['종합지수'].values, classes=3)
y = pd.Series(2, index=df.index)          # 2 = 비취약
y[df['종합지수'] <= t2] = 1               # 1 = 경계선
y[df['종합지수'] <= t1] = 0               # 0 = 취약

# 0-4. 설명변수(X): 응답ID, 종합지수, TYP, WT, Q11, Q26 빼기 (이유는 03번 참고)
drop = ['응답ID', '종합지수', 'TYP', 'WT'] + \
       [c for c in df.columns if c.startswith(('Q11', 'Q26'))]
X = df.drop(columns=drop)

# 0-5. train 80% / test 20%
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

# 03번의 최종 LightGBM과 같은 설정으로 학습
model = LGBMClassifier(n_estimators=300, learning_rate=0.03, max_depth=6, num_leaves=15,
                       min_child_samples=150, subsample=0.85, colsample_bytree=0.85,
                       reg_alpha=0.8, reg_lambda=4, class_weight='balanced',
                       random_state=0, n_jobs=1, verbose=-1, force_col_wise=True)
model.fit(train_x, train_y)


# 1. SHAP 값 계산 (test 데이터 기준)
explainer = shap.TreeExplainer(model)
sv = np.array(explainer.shap_values(test_x))
if sv.shape[-1] == 3:            # shap 버전에 따라 (행, 변수, 클래스) 모양으로 나옴
    sv = np.moveaxis(sv, -1, 0)  # → (클래스, 행, 변수)


# 2. 클래스별 평균 |SHAP| TOP 35 막대그래프
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
for ax, k, name in [(axes[0], 0, '취약'), (axes[1], 1, '경계선')]:
    imp = pd.Series(np.abs(sv[k]).mean(axis=0), index=test_x.columns) \
            .sort_values(ascending=False).head(35)
    ax.bar(imp.index, imp.values, color='#4C72B0')
    ax.set_title(f'SHAP 변수중요도 TOP35 - {name} 클래스 ({X.shape[1]}개 변수 기준 모델)')
    ax.set_ylabel('평균 |SHAP| 값')
    ax.tick_params(axis='x', rotation=90)
    print(f'\n[{name} 클래스 TOP 10]')
    print(imp.head(10).round(3))
plt.tight_layout()
plt.savefig('output/shap_bar_top35.png', dpi=150)
plt.close()


# 3. 취약 클래스 beeswarm (값이 클 때/작을 때 어느 방향으로 미는지)
#    빨강 = 변수값 높음, 파랑 = 낮음 / 오른쪽 = 취약 가능성 ↑
#    ※ 직업·거주지역은 순서 없는 코드값이라 색 해석에 주의
shap.summary_plot(sv[0], test_x, max_display=25, show=False)
plt.title('SHAP Summary - 취약 클래스')
plt.tight_layout()
plt.savefig('output/shap_beeswarm_취약.png', dpi=150)
plt.close()

print('\n그래프 저장 완료: output/shap_bar_top35.png, output/shap_beeswarm_취약.png')
