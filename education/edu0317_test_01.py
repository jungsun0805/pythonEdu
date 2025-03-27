#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import streamlit as st
import joblib
import seaborn as sns

# 데이터 로드 및 전처리 함수 구현
# 데이터 준비
plt.rcParams["font.family"] = 'Malgun Gothic'
plt.rcParams["axes.unicode_minus"] = False
pd.options.display.float_format = "{:.2f}".format
# 데이터셋: 'HR_comma_sep.csv' 파일을 사용합니다.
df = pd.read_csv("dataset/HR_comma_sep.csv")
df.head()


# In[8]:


# 주요 처리: 'Departments' 열 이름 수정 (공백 제거)
df.rename(columns={'Departments ':'Departments'},inplace=True)
# 범주형 변수('Departments', 'salary')를 One-Hot Encoding으로 변환
# drop_first=True 옵션을 사용하여 다중공선성 문제 방지
df = pd.get_dummies(df,columns=['Departments','salary'],drop_first=True)


# In[9]:


# 2. 특성 선택
# 선택된 특성:
# satisfaction_level: 직원 만족도
# number_project: 참여한 프로젝트 수
# time_spend_company: 회사 근무 기간(년)
X = df[['satisfaction_level','number_project','time_spend_company']]
# 타겟 변수: 'left' (퇴사 여부 - 0: 잔류, 1: 퇴사)
y = df[['left']]
# 데이터 분할 및 스케일링
# 데이터 분할: 훈련 데이터 80%, 테스트 데이터 20% (random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
# 데이터 스케일링: StandardScaler 사용
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[ ]:


# 3. 모델 선택 및 학습
# 모델: RandomForestClassifier
# 하이퍼파라미터:
# n_estimators=100 (트리 개수)
# random_state=42 (재현성 보장)
rf_model = RandomForestClassifier(random_state=42,n_estimators=100)
rf_model.fit(X_train_scaled,y_train)
joblib.dump(rf_model,'hr_model.pkl')
rf_pred = rf_model.predict(X_test_scaled)

feature_importance = pd.DataFrame({
    "특성": X.columns,
    "중요도":rf_model.feature_importances_
}).sort_values('중요도',ascending=False)

# 특성 중요도 시각화
plt.figure(figsize=(10, 6))
sns.barplot(x='중요도', y='특성', data=feature_importance)
plt.title('특성 중요도')
plt.show()
# In[ ]:


# Streamlit UI 구성
st.title("퇴사율 예측")
st.write("직원 만족도, 참여한 프로젝트 수, 회사 근무 기간(년)을 입력하고 퇴사율을 예측해보자")
# 사용자 입력 처리 및 예측 기능 구현
sl = st.slider("만족도",min_value=0.0,max_value=1.0,value=0.5)
nump = st.slider("프로젝트 수",min_value=0,max_value=10,value=3,step=1)
tsc = st.slider("근무 기간",min_value=0,max_value=30,value=5,step=1)
# 결과 표시 및 시각화
# 예측하기 버튼
if st.button('예측하기'):
    # 입력값을 모델에 전달
    rf_model = joblib.load('hr_model.pkl')
    input_data = np.array([[sl,nump,tsc]])
    prediction = rf_model.predict(input_data)[0]
    # 결과 출력
    if prediction == 1:
        st.error("예측 결과: 퇴사할 가능성이 높습니다.")
    else:
        st.success("예측 결과: 퇴사할 가능성이 낮습니다.")
    
    st.bar_chart(feature_importance.set_index("특성"))
