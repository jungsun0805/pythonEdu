#!/usr/bin/env python
# coding: utf-8

# In[107]:


# 데이터셋 설명
# Country name 나라명, Iso alpha ISO나라코드, year 년도, Happiness score 행복도점수, Log GDP per capita 1인당총생산
# Social support 사회지원, Healthy life expectancy at birth 기대수명, Freedom to make life choices 자유인식정도
# Generosity 관대함수준, Perceptions of corruption 부패에대한인식, Positive affect 긍정적 영향, Negative affect 부정적영향

#라이브러리 임포트
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns

# 폰트 지정
plt.rcParams["font.family"] = 'Malgun Gothic'
# 마이너스 깨짐 지정
plt.rcParams["axes.unicode_minus"] = False
# 실수표시 지정
pd.options.display.float_format = "{:.2f}".format

# 데이터셋 불러오기
file_path = "dataset/whr_200522.csv"
df = pd.read_csv(file_path, encoding="CP949")
#df.head()


# In[3]:


# In[4]:



# In[5]:


# 데이터 전처리
# 결측치 처리()
df['Log GDP per capita'] = df['Log GDP per capita'].fillna(df['Log GDP per capita'].mean())       # 1인당 GDP
df['Social support'] = df['Social support'].fillna(df['Social support'].mean())               # 사회적 지원
df['Healthy life expectancy at birth'] = df['Healthy life expectancy at birth'].fillna(df['Healthy life expectancy at birth'].mean())        # 기대수명
df['Freedom to make life choices'] = df['Freedom to make life choices'].fillna(df['Freedom to make life choices'].mean())               # 선택의자유
df['Generositys'] = df['Generosity'].fillna(df['Generosity'].mean())                      # 관대함수준
df['Perceptions of corruption'] = df['Perceptions of corruption'].fillna(df['Perceptions of corruption'].mean())                     # 부패의대한인식
df['Positive affect'] = df['Positive affect'].fillna(df['Positive affect'].mean())             # 긍정적영향
df['Negative affect'] = df['Negative affect'].fillna(df['Negative affect'].mean())             # 부정적영향

# 원-핫 인코딩
df_en = pd.get_dummies(df, columns=['Country name'],prefix='nn')


# In[63]:


# 모델
feature_names = ['year',
                 'Log GDP per capita',
                 'Social support',
                 'Healthy life expectancy at birth',
                 'Freedom to make life choices',
                 'Generosity',
                 'Perceptions of corruption',
                 'Positive affect',
                 'Negative affect'] + [col for col in df_en.columns if col.startswith('nn_')]
feature_names2 = ['year',
                 'Log GDP per capita',
                 'Social support',
                 'Healthy life expectancy at birth',
                 'Freedom to make life choices',
                 'Generosity',
                 'Perceptions of corruption',
                 'Positive affect',
                 'Negative affect']
X = df_en[feature_names2]
y = df_en['Happiness score']

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(random_state=42)

# 하이퍼파라미터 튜닝 GridSearchCV 사용
param_grid = {
    'n_estimators': [50,100,200,300],
    'max_depth':[4,6,8,10,12,14,16,18,20],
    'min_samples_split':[2,4,8,16]
}

# 모델 학습
# 최적의 모델 저장
#grid_search = GridSearchCV(estimator=rf_model,param_grid=param_grid,cv=3)
#grid_search.fit(X_train,y_train)
#best_rf_model = grid_search.best_estimator_
#print("Best Param:", grid_search.best_params_)
rf_model = RandomForestRegressor(random_state=42, n_estimators=300, max_depth=16, min_samples_split=2)
rf_model.fit(X_train,y_train)


# In[64]:


# 예측
rf_pred = rf_model.predict(X_test)
#rf_pred = best_rf_model.predict(X_test)

# 성능 평가
rf_mse = mean_squared_error(y_test,rf_pred)
rf_rmse = root_mean_squared_error(y_test,rf_pred)
rf_mae = mean_absolute_error(y_test,rf_pred)
rf_r2 = r2_score(y_test,rf_pred)


# In[21]:


# 년도별 행복 점수 시각화 함수
def plot_happiness_by_year(df, year_column, actual_column, predicted_column):
    plt.figure(figsize=(10, 6))
    
    df_grouped = df.groupby(year_column)[[actual_column, predicted_column]].mean()

    df_grouped.plot(kind='bar', figsize=(12, 6), alpha=0.7)
    plt.xlabel("년도")
    plt.ylabel("행복도")
    plt.title("실제값, 예측값 년도별 행복도 비교")
    plt.legend(["실제값", "예측값"])
    plt.xticks(rotation=45)
    plt.show()


# In[29]:


# 국가별 행복 점수 시각화 함수
def plot_happiness_by_country(df, country_column, year_column, actual_column, predicted_column, country_name):
    plt.figure(figsize=(10, 6))
    
    df_country = df[df[country_column] == country_name]
    df_country = df_country.sort_values(by=year_column)

    plt.plot(df_country[year_column], df_country[actual_column], marker='o', linestyle='-', label="실제값", color='blue')
    plt.plot(df_country[year_column], df_country[predicted_column], marker='s', linestyle='--', label="예측값", color='orange')

    plt.xlabel("년도")
    plt.ylabel("행복도")
    plt.title(f"{country_name}의 년도별 예측값, 실제값 비교")
    plt.legend()
    plt.grid(True)
    plt.show()


# In[ ]:
countrys = sorted(df['Country name'].unique())
st.title('국가 행복도 예측 프로그램')

# 탭 메뉴 설정
tab1, tab2, tab3 = st.tabs(["첫번째 국가 행복도 예측","두번째 국가 행복도 예측", "두 국가 행복도 비교"])

# 첫 번째 탭: 국가 행복도 예측
with tab1:
    st.header('국가 행복도 예측')

    # 특성 입력
    country1 = st.selectbox('나라를 선택하세요:', countrys, key="country1")
    year1 = st.number_input('년도', min_value=2024, max_value=2030, value=2025, step=1, key="year1")
    log_gdp1 = st.number_input('인당GDP', min_value=0.0, max_value=50.0, value=10.0, step=0.1, key="log_gdp1")
    social_support1 = st.number_input('사회적지원', min_value=0.0, max_value=1.0, value=0.9, step=0.01, key="social_support1")
    life_expectancy1 = st.number_input('기대수명', min_value=50, max_value=100, value=75, step=1, key="life_expectancy1")
    freedom1 = st.number_input('자유인식', min_value=0.0, max_value=1.0, value=0.8, step=0.01, key="freedom1")
    generosity1 = st.number_input('관대함', min_value=0.0, max_value=1.0, value=0.2, step=0.01, key="generosity1")
    corruption1 = st.number_input('부패에대한인식', min_value=0.0, max_value=1.0, value=0.1, step=0.01, key="corruption1")
    positive_affect1 = st.number_input('긍정적영향', min_value=0.0, max_value=1.0, value=0.8, step=0.01, key="positive_affect1")
    negative_affect1 = st.number_input('부정적영향', min_value=0.0, max_value=1.0, value=0.1, step=0.01, key="negative_affect1")

    # 입력값을 DataFrame으로 변환
    input_data = pd.DataFrame({
        'year': [year1],
        'Log GDP per capita': [log_gdp1],
        'Social support': [social_support1],
        'Healthy life expectancy at birth': [life_expectancy1],
        'Freedom to make life choices': [freedom1],
        'Generosity': [generosity1],
        'Perceptions of corruption': [corruption1],
        'Positive affect': [positive_affect1],
        'Negative affect': [negative_affect1]
    })

    # 예측 수행
    # 예측하기 버튼
    #if st.button('예측하기', key='button1'):
    predicted_happiness = rf_model.predict(input_data)

    st.write(f'{country1}의 예측된 행복도 점수: {predicted_happiness[0]:.2f}')
with tab2:
    st.header('국가 행복도 예측')

    # 특성 입력
    country2 = st.selectbox('두 번째 나라를 선택하세요:', countrys, key="country2")
    year2 = st.number_input('년도', min_value=2024, max_value=2030, value=2025, step=1, key="year2")
    log_gdp2 = st.number_input('인당GDP', min_value=0.0, max_value=50.0, value=10.0, step=0.1, key="log_gdp2")
    social_support2 = st.number_input('사회적지원', min_value=0.0, max_value=1.0, value=0.9, step=0.01, key="social_support2")
    life_expectancy2 = st.number_input('기대수명', min_value=50, max_value=100, value=75, step=1, key="life_expectancy2")
    freedom2 = st.number_input('자유인식', min_value=0.0, max_value=1.0, value=0.8, step=0.01, key="freedom2")
    generosity2 = st.number_input('관대함', min_value=0.0, max_value=1.0, value=0.2, step=0.01, key="generosity2")
    corruption2 = st.number_input('부패에대한인식', min_value=0.0, max_value=1.0, value=0.1, step=0.01, key="corruption2")
    positive_affect2 = st.number_input('긍정적영향', min_value=0.0, max_value=1.0, value=0.8, step=0.01, key="positive_affect2")
    negative_affect2 = st.number_input('부정적영향', min_value=0.0, max_value=1.0, value=0.1, step=0.01, key="negative_affect2")

    # 입력값을 DataFrame으로 변환
    input_data2 = pd.DataFrame({
        'year': [year2],
        'Log GDP per capita': [log_gdp2],
        'Social support': [social_support2],
        'Healthy life expectancy at birth': [life_expectancy2],
        'Freedom to make life choices': [freedom2],
        'Generosity': [generosity2],
        'Perceptions of corruption': [corruption2],
        'Positive affect': [positive_affect2],
        'Negative affect': [negative_affect2]
    })

    # 예측 수행
    #if st.button('예측하기', key='button2'):
    predicted_happiness2 = rf_model.predict(input_data2)

    st.write(f'{country2}의 예측된 행복도 점수: {predicted_happiness2[0]:.2f}')
# 두 번째 탭: 두 국가 행복도 비교
with tab3:
    st.header('두 국가 행복도 비교')

    # 두 나라 행복도 비교 시각화
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=[country1, country2], y=[predicted_happiness[0], predicted_happiness2[0]], palette='viridis', ax=ax)
    ax.set_title('두 나라의 행복도 비교', fontsize=14)
    ax.set_ylabel('행복도 점수', fontsize=12)
    ax.set_xlabel('나라', fontsize=12)
    st.pyplot(fig)
