# 03. 모델 비교 → 최종 모델 → 중요 변수 TOP 35
# 입력: output/디지털취약계층_X변수.xlsx (01번), output/Y변수_종합지수.csv (02번)
#
# 평가 기준 3가지
#   macro_f1       : 세 집단을 골고루 잘 맞히는지 (가장 중요)
#   취약_recall     : 실제 취약층 중 몇 %를 찾아냈는지 (놓치면 안 되니까)
#   train-test 차이 : train만 잘 맞히고 test는 못 맞히는 '과적합'인지

import warnings
import numpy as np
import pandas as pd
from skimage.filters import threshold_multiotsu
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, f1_score, recall_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# 0. 데이터 준비
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

# 0-4. 설명변수(X): 아래 열은 빼기
#      응답ID, 종합지수 : 번호와 정답 자체
#      TYP : 계층 라벨 (라벨 없이 역량으로 분류하는 게 목적)
#      WT  : 통계용 가중치
#      Q11 : 인터넷 이용 일수 → Q10과 내용 겹침
#      Q26 : AI 역량 → Y와 비슷한 문항
drop = ['응답ID', '종합지수', 'TYP', 'WT'] + \
       [c for c in df.columns if c.startswith(('Q11', 'Q26'))]
X = df.drop(columns=drop)

# 0-5. train 80% / test 20%, 세 집단 비율은 똑같이 유지(stratify)
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
print('변수 개수:', X.shape[1])


def evaluate(name, model, tr_x, te_x):
    """모델 점수 3가지 계산"""
    pred_test = model.predict(te_x)
    test_acc = accuracy_score(test_y, pred_test)
    train_acc = accuracy_score(train_y, model.predict(tr_x))
    return {'모델': name,
            'macro_f1': round(f1_score(test_y, pred_test, average='macro'), 4),
            '취약_recall': round(recall_score(test_y, pred_test, labels=[0], average='macro'), 4),
            'train-test차이': round(train_acc - test_acc, 4)}


# 1. 기본 모델 3종 (기본 설정 그대로)
rf = RandomForestClassifier(random_state=0).fit(train_x, train_y)
lgb = LGBMClassifier(random_state=0, verbose=-1).fit(train_x, train_y)
xgb = XGBClassifier(random_state=0, eval_metric='mlogloss').fit(train_x, train_y)

print(pd.DataFrame([evaluate('RF', rf, train_x, test_x),
                    evaluate('LGB', lgb, train_x, test_x),
                    evaluate('XGB', xgb, train_x, test_x)]))
# → train-test 차이가 커서 과적합 → 튜닝 필요

# 2. 튜닝 (GridSearchCV: 설정 조합을 다 해보고 macro_f1이 가장 높은 것 고르기)
#    cv=5 → train을 5조각으로 나눠 돌아가며 검증
gs_rf = GridSearchCV(
    RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=0),
    {'min_samples_split': range(2, 21), 'max_features': ['sqrt', 'log2']},
    cv=5, scoring='f1_macro', n_jobs=-1)
gs_rf.fit(train_x, train_y)

gs_lgb = GridSearchCV(
    LGBMClassifier(n_estimators=200, class_weight='balanced', random_state=0, verbose=-1),
    {'max_depth': range(3, 11), 'num_leaves': range(20, 51, 10)},
    cv=5, scoring='f1_macro', n_jobs=-1)
gs_lgb.fit(train_x, train_y)

weight = compute_sample_weight('balanced', train_y)   # XGBoost는 집단 크기 보정을 이렇게 따로 넣음
gs_xgb = GridSearchCV(
    XGBClassifier(learning_rate=0.1, random_state=0, eval_metric='mlogloss'),
    {'max_depth': range(3, 9), 'n_estimators': range(100, 301, 100)},
    cv=5, scoring='f1_macro', n_jobs=-1)
gs_xgb.fit(train_x, train_y, sample_weight=weight)

print(pd.DataFrame([evaluate('RF_튜닝', gs_rf.best_estimator_, train_x, test_x),
                    evaluate('LGB_튜닝', gs_lgb.best_estimator_, train_x, test_x),
                    evaluate('XGB_튜닝', gs_xgb.best_estimator_, train_x, test_x)]))
# → 성능이 비슷해서 LightGBM을 최종 모델로 선택하고 과적합을 더 줄이기

# 3. 최종 모델: LightGBM (과적합을 줄이는 설정)
LGBM_PARAMS = dict(
    n_estimators=300,         # 나무 300개
    learning_rate=0.03,       # 한 번에 조금씩만 배우기 (과적합 방지)
    max_depth=6,              # 나무 깊이 최대 6단계
    num_leaves=15,            # 나무 하나의 잎 최대 15개
    min_child_samples=150,    # 잎 하나에 최소 150명 (너무 잘게 쪼개지 않기)
    subsample=0.85,           # 나무마다 사람 85%만 뽑아서 학습
    colsample_bytree=0.85,    # 나무마다 변수 85%만 뽑아서 학습
    reg_alpha=0.8, reg_lambda=4,   # 규제 (모델이 너무 복잡해지지 않게)
    class_weight='balanced',  # 집단 크기 차이 보정 (취약층이 적어도 똑같이 중요하게)
    random_state=0, n_jobs=1, verbose=-1, force_col_wise=True,
)
final = LGBMClassifier(**LGBM_PARAMS).fit(train_x, train_y)
print(evaluate('LGB_최종', final, train_x, test_x))
# → 성능은 조금 내려가도 과적합이 크게 줄고, 취약층은 더 잘 찾음

# 교차검증: 데이터를 5번 다르게 나눠서 돌려도 점수가 비슷한지 확인
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
score = cross_validate(LGBMClassifier(**LGBM_PARAMS), X, y, cv=cv,
                       scoring='f1_macro', return_train_score=True)
print('교차검증 macro_f1  train:', round(score['train_score'].mean(), 4),
      '/ test:', round(score['test_score'].mean(), 4))

# 4. 중요 변수 TOP 35 (최종 모델 기준)
#    중요도 = 최종 모델이 나무를 나눌 때 그 변수를 쓴 횟수
importance = pd.Series(final.feature_importances_, index=train_x.columns).sort_values(ascending=False)
top35 = importance.head(35)
print(top35)

# 해석
#   인구통계(연령, 거주지역, 직업, 가구소득, 학력)가 상위권
#   문제해결 방식(Q5_1 스스로 해결, Q5_4 전문인력 도움)
#   디지털 효능감(Q22 문항)
#   → 디지털 취약성은 인구통계뿐 아니라 개인의 문제해결 방식과 디지털 자신감의 영향을 크게 받음
