# 05. 의사결정나무
# 입력: output/디지털취약계층_X변수.xlsx (01번), output/Y변수_종합지수.csv (02번)
# 출력: output/decision_tree.png
#
# '취약' vs '경계선' 두 집단만 놓고, 어떤 조건에서 갈리는지 규칙으로 확인 (비취약 집단은 제외)

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.filters import threshold_multiotsu
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import (accuracy_score, f1_score, recall_score, roc_auc_score)

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'Malgun Gothic'   # Mac: 'AppleGothic'
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


# 1. 데이터: 취약(0) + 경계선(1)만 사용, 타겟은 '취약이면 1'
mask = y.isin([0, 1])
X = X[mask]
y = (y[mask] == 0).astype(int)        # 1 = 취약, 0 = 경계선
print('대상:', len(y), '명 / 기준 취약률:', round(y.mean() * 100, 1), '%')

# 의사결정나무는 결측을 못 받으므로 중앙값으로 채움
X = X.fillna(X.median())

train_x, test_x, train_y, test_y = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y)


# 2. 모델: 해석하기 쉽게 깊이 4, 한 잎(leaf)에 최소 100명
tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=100, random_state=0)
tree.fit(train_x, train_y)

pred = tree.predict(test_x)
recall = recall_score(test_y, pred, average=None)
print('Accuracy       :', round(accuracy_score(test_y, pred), 3))
print('Macro F1       :', round(f1_score(test_y, pred, average='macro'), 3))
print('취약 Recall     :', round(recall[1], 3))
print('경계선 Recall   :', round(recall[0], 3))
print('AUC            :', round(roc_auc_score(test_y, tree.predict_proba(test_x)[:, 1]), 3))
print('Train-Test 차이 :', round(accuracy_score(train_y, tree.predict(train_x))
                              - accuracy_score(test_y, pred), 3))


# 3. 규칙 확인용 나무: 같은 설정으로 두 집단 전체에 다시 학습해서 그림으로 그림
tree_all = DecisionTreeClassifier(max_depth=4, min_samples_leaf=100, random_state=0)
tree_all.fit(X, y)
print(export_text(tree_all, feature_names=list(X.columns)))

# 잎(최종 그룹)마다 사람 수와 취약률
leaf = pd.DataFrame({'leaf': tree_all.apply(X), '취약': y})
leaf_summary = leaf.groupby('leaf')['취약'].agg(인원='size', 취약률='mean')
leaf_summary['취약률'] = (leaf_summary['취약률'] * 100).round(1)
print(leaf_summary)

plt.figure(figsize=(28, 12))
plot_tree(tree_all, feature_names=X.columns, class_names=['경계선', '취약'],
          filled=True, rounded=True, fontsize=9, proportion=False)
plt.tight_layout()
plt.savefig('output/decision_tree.png', dpi=150)
print('그래프 저장 완료: output/decision_tree.png')
