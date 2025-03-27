import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# 파일 업로드 기능 추가
st.title("국가별 행복도 시각화")
df = pd.read_csv('dataset/whr_200522.csv', encoding="CP949")

# LadderScore만 따로 스케일링
ladder_scaler = MinMaxScaler()
ladder_scaler.fit_transform(df[['LadderScore']])  # 원래 행복 점수 데이터로 스케일러 학습

tab1, tab2 = st.tabs(["행복도 예측(파일업로드)", "연도별 행복도 세계지도"])

# 한글
plt.rcParams['font.family'] = 'Malgun Gothic'
# 마이너스 깨짐 처리
plt.rcParams['axes.unicode_minus'] = False
# 숫자
pd.options.display.float_format = "{:.2f}".format

pred_df = pd.read_csv('dataset/whr_temp.csv', encoding='cp949')

# 전처리(결측치 값 평균값으로 대체)
pred_df['Economy'] = pred_df['Economy'].fillna(0)
pred_df['HealthyLife'] = pred_df['HealthyLife'].fillna(0)
pred_df['SocialSupport'] = pred_df['SocialSupport'].fillna(0)
pred_df['Freedom'] = pred_df['Freedom'].fillna(0)

with tab1:
    # 파일 업로드
    uploaded_file = st.file_uploader("행복도파일을 업로드하세요(.csv)", type="csv", key="file1")

    if uploaded_file is not None:        
        # 데이터 읽기
        data = pd.read_csv(uploaded_file)
        
        data['Economy'] = data['Economy'].fillna(data['Economy'].mean())
        data['HealthyLife'] = data['HealthyLife'].fillna(data['HealthyLife'].mean())
        data['SocialSupport'] = data['SocialSupport'].fillna(data['SocialSupport'].mean())
        data['Freedom'] = data['Freedom'].fillna(data['Freedom'].mean())

        # 데이터 미리보기
        st.write("데이터 Preview:")
        st.write(data.head())

        # 국가 리스트 선택
        countries = st.multiselect(
            "국가를 선택하세요", 
            options=data['CountryName'].unique(),
            default=[]
        )
        if len(countries) > 0:
            # 선택한 국가 데이터 필터링
            filtered_data = data[data['CountryName'].isin(countries)]

            # 모델 데이터 읽기
            model = load_model("model/whr_gru_model.h5")

            X_val_list = []
            for country in countries:
                country_data = filtered_data[filtered_data['CountryName'] == country]
                
                # 필요한 컬럼만 선택
                X_val = country_data.drop(['Year', 'CountryName', 'Generosity', 'Corruption'], axis=1).values  # (time_steps, features)
                #st.write(X_val)
                # Feature 수가 4개면 1개 추가하여 (time_steps, 5)로 맞춤
                if X_val.shape[1] == 4:
                    X_val = np.hstack([np.zeros((X_val.shape[0], 1)), X_val])  # (time_steps, 5)
                X_val_list.append(X_val)

            # 리스트를 NumPy 배열로 변환 (차원이 맞는지 확인)
            X_val = np.array(X_val_list)  # (num_countries, time_steps, features)

            # 차원 변경 (모델 입력 형식에 맞게)
            X_val = np.stack(X_val)  # (num_countries, time_steps, features)
            # 예측 수행
            y_pred = model.predict(X_val, verbose=0)
            # 역정규화
            y_pred_invers = ladder_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

            # 예측을 session_state에 저장하여 추후 활용
            st.session_state['y_pred'] = y_pred_invers
            st.session_state['filtered_data'] = filtered_data

            # Happiness Score Time Series Plot (연도별 행복도)
            st.subheader("검색 국가별 행복도")
            fig, ax = plt.subplots(figsize=(10, 6))
            for i, country in enumerate(countries):
                country_data = filtered_data[filtered_data['CountryName'] == country]

                # 국가별 예측값 추출 (time_steps, 1) → (time_steps,)
                y_pred_country = y_pred_invers[i].flatten()  
                #st.write("예측값: ",y_pred_country[0])

                # 해당 국가와 2023년의 데이터가 존재하는지 확인
                condition = (pred_df['CountryName'] == country) & (pred_df['Year'] == 2023)
                if pred_df[condition].empty:
                    # 새로운 데이터가 없으면 추가
                    new_data = pd.DataFrame({
                        "CountryName": [country],
                        "Year": [2023],
                        "LadderScore": [y_pred_country[0]],  # 예측한 첫 번째 값
                        "Economy": [country_data['Economy'].values[0]],
                        "SocialSupport": [country_data['SocialSupport'].values[0]],
                        "HealthyLife": [country_data['HealthyLife'].values[0]],
                        "Freedom": [country_data['Freedom'].values[0]],
                        "Generosity": [country_data['Generosity'].values[0]],
                        "Corruption": [country_data['Corruption'].values[0]]
                    })
                    pred_df = pd.concat([pred_df, new_data], ignore_index=True)
                else:
                    # 이미 존재하는 데이터라면 수정
                    pred_df.loc[condition, 'LadderScore'] = y_pred_country[0]

                # CSV 파일로 저장
                pred_df.to_csv('dataset/whr_temp.csv', index=False, encoding='cp949')
                view_df = pd.DataFrame({
                    "CountryName": [country],
                    "Year": [2023],
                    "LadderScore": [y_pred_country[0]],  # 예측한 첫 번째 값
                    "Economy": [country_data['Economy'].values[0]],
                    "SocialSupport": [country_data['SocialSupport'].values[0]],
                    "HealthyLife": [country_data['HealthyLife'].values[0]],
                    "Freedom": [country_data['Freedom'].values[0]],
                    "Generosity": [country_data['Generosity'].values[0]],
                    "Corruption": [country_data['Corruption'].values[0]]
                })
                
                sns.barplot(x='CountryName', y='LadderScore', data=view_df, ax=ax, palette="viridis")
            ax.set_title("행복도")
            ax.set_xlabel("")
            ax.set_ylabel("행복도")
            ax.legend()
            st.pyplot(fig)
            
            # Happiness Score Time Series Plot (연도별 행복도)
            st.subheader("검색 국가별, 연도별 행복도")
            fig, ax = plt.subplots(figsize=(10, 6))
            for country in countries:
                country_data = pred_df[pred_df['CountryName'] == country]
                
                # 국가별 연도별 행복도 선 그래프
                sns.lineplot(x='Year', y='LadderScore', data=country_data, label=country, ax=ax, marker='o')
            
            # 그래프 제목 및 축 레이블
            ax.set_title("국가별 연도별 행복도 변화")
            ax.set_xlabel("연도")
            ax.set_ylabel("행복도")
            # 범례 표시
            ax.legend(title="국가")
            # 그래프 출력
            st.pyplot(fig)

            # 2023년 데이터를 기준으로 상위 10개 국가의 행복도 순위 (Bar Chart)
            st.subheader("2023년 상위 10 행복도 순위")
            
            # 2023년 데이터만 필터링
            top_10_happy_2023 = pred_df[pred_df['Year'] == 2023][['CountryName', 'LadderScore']].sort_values(by='LadderScore', ascending=False).head(10)

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='LadderScore', y='CountryName', data=top_10_happy_2023, ax=ax, palette="viridis")
            ax.set_title("2023년 Top 10 행복도 순위")
            ax.set_xlabel("행복도")
            ax.set_ylabel("국가")
            st.pyplot(fig)
            
            happy_2023 = pred_df[pred_df['Year'] == 2023][['CountryName', 'LadderScore','Economy','SocialSupport']]

            # GDP Per Capita vs Happiness Score (산점도)
            st.subheader("경제적 요인 vs 행복도")
            fig = px.scatter(happy_2023, x='Economy', y='LadderScore', color='CountryName', title="경제적 요인 vs 행복도 산점도")
            st.plotly_chart(fig)

            # Social Support vs Happiness Score (산점도)
            st.subheader("사회적 지원 vs 행복도")
            fig = px.scatter(happy_2023, x='SocialSupport', y='LadderScore', color='CountryName', title="사회적 지원 vs 행복도 산점도")
            st.plotly_chart(fig)
    else:
        st.write("행복도 파일을 업로드 하세요(.csv)")

with tab2:
    # 연도 선택
    selected_year = st.selectbox("연도를 선택하세요", options=sorted(pred_df['Year'].unique(), reverse=True), index=0)

    # 선택된 연도에 맞는 데이터 필터링
    year_filtered_data = pred_df[pred_df['Year'] == selected_year]
    
    # 지도 시각화
    st.subheader("행복도 분포 세계지도")

    # Plotly Express로 지도 시각화
    fig = px.choropleth(data_frame=year_filtered_data,
                        locations="CountryName",  # 국가 열
                        locationmode="country names",  # 국가 이름으로 위치 지정
                        color="LadderScore",  # 행복도 컬럼
                        color_continuous_scale=px.colors.sequential.Plasma,  # 색상 팔레트
                        title=f"{selected_year}년 행복도 분포")

    # 지도 출력
    st.plotly_chart(fig)