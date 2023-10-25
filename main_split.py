# 데이터 불러오기
import pandas as pd
df = pd.read_csv('Dataset/Seoul_House_Price.csv', encoding='cp949', low_memory=False)
# 필요 없는 칼럼 삭제
df.drop(columns=['접수연도', '자치구코드', '법정동코드', '법정동명', '지번구분', '지번구분명', '본번', '부번', '건물명', '건물용도', '계약일', '토지면적(㎡)', '권리구분', '취소일', '건축년도', '신고구분', '신고한 개업공인중개사 시군구명'], inplace=True)
# print(df)

# 결과값과 입력값 분리
Y = df['물건금액(만원)']
X = df.drop(columns=['물건금액(만원)'])
print('Data Prepared')

# 훈련 데이터와 테스트 데이터 분리
from sklearn.model_selection import train_test_split
train_score_sum = 0
test_score_sum = 0

# 강남구 다중회귀
from sklearn.linear_model import LinearRegression
model_Gangnam = LinearRegression()
X_Gangnam = X[df['자치구명'] == '강남구'].drop(columns=['자치구명']).to_numpy()
Y_Gangnam = Y[df['자치구명'] == '강남구'].to_numpy()
X_Gangnam_train, X_Gangnam_test, Y_Gangnam_train, Y_Gangnam_test = train_test_split(X_Gangnam, Y_Gangnam, test_size=0.2)
print('강남구 Model Learning Start')
model_Gangnam.fit(X_Gangnam_train, Y_Gangnam_train)
print('강남구 Model Learning End')
print('강남구 Train Score :', model_Gangnam.score(X_Gangnam_train, Y_Gangnam_train))
print('강남구 Test Score :', model_Gangnam.score(X_Gangnam_test, Y_Gangnam_test))
train_score_sum += model_Gangnam.score(X_Gangnam_train, Y_Gangnam_train)
test_score_sum += model_Gangnam.score(X_Gangnam_test, Y_Gangnam_test)

# 강동구 다중회귀
model_Gangdong = LinearRegression()
X_Gangdong = X[df['자치구명'] == '강동구'].drop(columns=['자치구명']).to_numpy()
Y_Gangdong = Y[df['자치구명'] == '강동구'].to_numpy()
X_Gangdong_train, X_Gangdong_test, Y_Gangdong_train, Y_Gangdong_test = train_test_split(X_Gangdong, Y_Gangdong, test_size=0.2)
print('강동구 Model Learning Start')
model_Gangdong.fit(X_Gangdong_train, Y_Gangdong_train)
print('강동구 Model Learning End')
print('강동구 Train Score :', model_Gangdong.score(X_Gangdong_train, Y_Gangdong_train))
print('강동구 Test Score :', model_Gangdong.score(X_Gangdong_test, Y_Gangdong_test))
train_score_sum += model_Gangdong.score(X_Gangdong_train, Y_Gangdong_train)
test_score_sum += model_Gangdong.score(X_Gangdong_test, Y_Gangdong_test)

# 강북구 다중회귀
model_Gangbuk = LinearRegression()
X_Gangbuk = X[df['자치구명'] == '강북구'].drop(columns=['자치구명']).to_numpy()
Y_Gangbuk = Y[df['자치구명'] == '강북구'].to_numpy()
X_Gangbuk_train, X_Gangbuk_test, Y_Gangbuk_train, Y_Gangbuk_test = train_test_split(X_Gangbuk, Y_Gangbuk, test_size=0.2)
print('강북구 Model Learning Start')
model_Gangbuk.fit(X_Gangbuk_train, Y_Gangbuk_train)
print('강북구 Model Learning End')
print('강북구 Train Score :', model_Gangbuk.score(X_Gangbuk_train, Y_Gangbuk_train))
print('강북구 Test Score :', model_Gangbuk.score(X_Gangbuk_test, Y_Gangbuk_test))
train_score_sum += model_Gangbuk.score(X_Gangbuk_train, Y_Gangbuk_train)
test_score_sum += model_Gangbuk.score(X_Gangbuk_test, Y_Gangbuk_test)

# 강서구 다중회귀
model_Gangseo = LinearRegression()
X_Gangseo = X[df['자치구명'] == '강서구'].drop(columns=['자치구명']).to_numpy()
Y_Gangseo = Y[df['자치구명'] == '강서구'].to_numpy()
X_Gangseo_train, X_Gangseo_test, Y_Gangseo_train, Y_Gangseo_test = train_test_split(X_Gangseo, Y_Gangseo, test_size=0.2)
print('강서구 Model Learning Start')
model_Gangseo.fit(X_Gangseo_train, Y_Gangseo_train)
print('강서구 Model Learning End')
print('강서구 Train Score :', model_Gangseo.score(X_Gangseo_train, Y_Gangseo_train))
print('강서구 Test Score :', model_Gangseo.score(X_Gangseo_test, Y_Gangseo_test))
train_score_sum += model_Gangseo.score(X_Gangseo_train, Y_Gangseo_train)
test_score_sum += model_Gangseo.score(X_Gangseo_test, Y_Gangseo_test)

# 관악구 다중회귀
model_Gwanak = LinearRegression()
X_Gwanak = X[df['자치구명'] == '관악구'].drop(columns=['자치구명']).to_numpy()
Y_Gwanak = Y[df['자치구명'] == '관악구'].to_numpy()
X_Gwanak_train, X_Gwanak_test, Y_Gwanak_train, Y_Gwanak_test = train_test_split(X_Gwanak, Y_Gwanak, test_size=0.2)
print('관악구 Model Learning Start')
model_Gwanak.fit(X_Gwanak_train, Y_Gwanak_train)
print('관악구 Model Learning End')
print('관악구 Train Score :', model_Gwanak.score(X_Gwanak_train, Y_Gwanak_train))
print('관악구 Test Score :', model_Gwanak.score(X_Gwanak_test, Y_Gwanak_test))
train_score_sum += model_Gwanak.score(X_Gwanak_train, Y_Gwanak_train)
test_score_sum += model_Gwanak.score(X_Gwanak_test, Y_Gwanak_test)

# 광진구 다중회귀
model_Gwangjin = LinearRegression()
X_Gwangjin = X[df['자치구명'] == '광진구'].drop(columns=['자치구명']).to_numpy()
Y_Gwangjin = Y[df['자치구명'] == '광진구'].to_numpy()
X_Gwangjin_train, X_Gwangjin_test, Y_Gwangjin_train, Y_Gwangjin_test = train_test_split(X_Gwangjin, Y_Gwangjin, test_size=0.2)
print('광진구 Model Learning Start')
model_Gwangjin.fit(X_Gwangjin_train, Y_Gwangjin_train)
print('광진구 Model Learning End')
print('광진구 Train Score :', model_Gwangjin.score(X_Gwangjin_train, Y_Gwangjin_train))
print('광진구 Test Score :', model_Gwangjin.score(X_Gwangjin_test, Y_Gwangjin_test))
train_score_sum += model_Gwangjin.score(X_Gwangjin_train, Y_Gwangjin_train)
test_score_sum += model_Gwangjin.score(X_Gwangjin_test, Y_Gwangjin_test)

# 구로구 다중회귀
model_Guro = LinearRegression()
X_Guro = X[df['자치구명'] == '구로구'].drop(columns=['자치구명']).to_numpy()
Y_Guro = Y[df['자치구명'] == '구로구'].to_numpy()
X_Guro_train, X_Guro_test, Y_Guro_train, Y_Guro_test = train_test_split(X_Guro, Y_Guro, test_size=0.2)
print('구로구 Model Learning Start')
model_Guro.fit(X_Guro_train, Y_Guro_train)
print('구로구 Model Learning End')
print('구로구 Train Score :', model_Guro.score(X_Guro_train, Y_Guro_train))
print('구로구 Test Score :', model_Guro.score(X_Guro_test, Y_Guro_test))
train_score_sum += model_Guro.score(X_Guro_train, Y_Guro_train)
test_score_sum += model_Guro.score(X_Guro_test, Y_Guro_test)

# 금천구 다중회귀
model_Geumcheon = LinearRegression()
X_Geumcheon = X[df['자치구명'] == '금천구'].drop(columns=['자치구명']).to_numpy()
Y_Geumcheon = Y[df['자치구명'] == '금천구'].to_numpy()
X_Geumcheon_train, X_Geumcheon_test, Y_Geumcheon_train, Y_Geumcheon_test = train_test_split(X_Geumcheon, Y_Geumcheon, test_size=0.2)
print('금천구 Model Learning Start')
model_Geumcheon.fit(X_Geumcheon_train, Y_Geumcheon_train)
print('금천구 Model Learning End')
print('금천구 Train Score :', model_Geumcheon.score(X_Geumcheon_train, Y_Geumcheon_train))
print('금천구 Test Score :', model_Geumcheon.score(X_Geumcheon_test, Y_Geumcheon_test))
train_score_sum += model_Geumcheon.score(X_Geumcheon_train, Y_Geumcheon_train)
test_score_sum += model_Geumcheon.score(X_Geumcheon_test, Y_Geumcheon_test)

# 노원구 다중회귀
model_Nowon = LinearRegression()
X_Nowon = X[df['자치구명'] == '노원구'].drop(columns=['자치구명']).to_numpy()
Y_Nowon = Y[df['자치구명'] == '노원구'].to_numpy()
X_Nowon_train, X_Nowon_test, Y_Nowon_train, Y_Nowon_test = train_test_split(X_Nowon, Y_Nowon, test_size=0.2)
print('노원구 Model Learning Start')
model_Nowon.fit(X_Nowon_train, Y_Nowon_train)
print('노원구 Model Learning End')
print('노원구 Train Score :', model_Nowon.score(X_Nowon_train, Y_Nowon_train))
print('노원구 Test Score :', model_Nowon.score(X_Nowon_test, Y_Nowon_test))
train_score_sum += model_Nowon.score(X_Nowon_train, Y_Nowon_train)
test_score_sum += model_Nowon.score(X_Nowon_test, Y_Nowon_test)

# 도봉구 다중회귀
model_Dobong = LinearRegression()
X_Dobong = X[df['자치구명'] == '도봉구'].drop(columns=['자치구명']).to_numpy()
Y_Dobong = Y[df['자치구명'] == '도봉구'].to_numpy()
X_Dobong_train, X_Dobong_test, Y_Dobong_train, Y_Dobong_test = train_test_split(X_Dobong, Y_Dobong, test_size=0.2)
print('도봉구 Model Learning Start')
model_Dobong.fit(X_Dobong_train, Y_Dobong_train)
print('도봉구 Model Learning End')
print('도봉구 Train Score :', model_Dobong.score(X_Dobong_train, Y_Dobong_train))
print('도봉구 Test Score :', model_Dobong.score(X_Dobong_test, Y_Dobong_test))
train_score_sum += model_Dobong.score(X_Dobong_train, Y_Dobong_train)
test_score_sum += model_Dobong.score(X_Dobong_test, Y_Dobong_test)

# 동대문구 다중회귀
model_Dongdaemun = LinearRegression()
X_Dongdaemun = X[df['자치구명'] == '동대문구'].drop(columns=['자치구명']).to_numpy()
Y_Dongdaemun = Y[df['자치구명'] == '동대문구'].to_numpy()
X_Dongdaemun_train, X_Dongdaemun_test, Y_Dongdaemun_train, Y_Dongdaemun_test = train_test_split(X_Dongdaemun, Y_Dongdaemun, test_size=0.2)
print('동대문구 Model Learning Start')
model_Dongdaemun.fit(X_Dongdaemun_train, Y_Dongdaemun_train)
print('동대문구 Model Learning End')
print('동대문구 Train Score :', model_Dongdaemun.score(X_Dongdaemun_train, Y_Dongdaemun_train))
print('동대문구 Test Score :', model_Dongdaemun.score(X_Dongdaemun_test, Y_Dongdaemun_test))
train_score_sum += model_Dongdaemun.score(X_Dongdaemun_train, Y_Dongdaemun_train)
test_score_sum += model_Dongdaemun.score(X_Dongdaemun_test, Y_Dongdaemun_test)

# 동작구 다중회귀
model_Dongjak = LinearRegression()
X_Dongjak = X[df['자치구명'] == '동작구'].drop(columns=['자치구명']).to_numpy()
Y_Dongjak = Y[df['자치구명'] == '동작구'].to_numpy()
X_Dongjak_train, X_Dongjak_test, Y_Dongjak_train, Y_Dongjak_test = train_test_split(X_Dongjak, Y_Dongjak, test_size=0.2)
print('동작구 Model Learning Start')
model_Dongjak.fit(X_Dongjak_train, Y_Dongjak_train)
print('동작구 Model Learning End')
print('동작구 Train Score :', model_Dongjak.score(X_Dongjak_train, Y_Dongjak_train))
print('동작구 Test Score :', model_Dongjak.score(X_Dongjak_test, Y_Dongjak_test))
train_score_sum += model_Dongjak.score(X_Dongjak_train, Y_Dongjak_train)
test_score_sum += model_Dongjak.score(X_Dongjak_test, Y_Dongjak_test)

# 마포구 다중회귀
model_Mapo = LinearRegression()
X_Mapo = X[df['자치구명'] == '마포구'].drop(columns=['자치구명'])
Y_Mapo = Y[df['자치구명'] == '마포구']
X_Mapo_train, X_Mapo_test, Y_Mapo_train, Y_Mapo_test = train_test_split(X_Mapo, Y_Mapo, test_size=0.2)
print('마포구 Model Learning Start')
model_Mapo.fit(X_Mapo_train, Y_Mapo_train)
print('마포구 Model Learning End')
print('마포구 Train Score :', model_Mapo.score(X_Mapo_train, Y_Mapo_train))
print('마포구 Test Score :', model_Mapo.score(X_Mapo_test, Y_Mapo_test))
train_score_sum += model_Mapo.score(X_Mapo_train, Y_Mapo_train)
test_score_sum += model_Mapo.score(X_Mapo_test, Y_Mapo_test)

# 서대문구 다중회귀
model_Seodaemun = LinearRegression()
X_Seodaemun = X[df['자치구명'] == '서대문구'].drop(columns=['자치구명'])
Y_Seodaemun = Y[df['자치구명'] == '서대문구']
X_Seodaemun_train, X_Seodaemun_test, Y_Seodaemun_train, Y_Seodaemun_test = train_test_split(X_Seodaemun, Y_Seodaemun, test_size=0.2)
print('서대문구 Model Learning Start')
model_Seodaemun.fit(X_Seodaemun_train, Y_Seodaemun_train)
print('서대문구 Model Learning End')
print('서대문구 Train Score :', model_Seodaemun.score(X_Seodaemun_train, Y_Seodaemun_train))
print('서대문구 Test Score :', model_Seodaemun.score(X_Seodaemun_test, Y_Seodaemun_test))
train_score_sum += model_Seodaemun.score(X_Seodaemun_train, Y_Seodaemun_train)
test_score_sum += model_Seodaemun.score(X_Seodaemun_test, Y_Seodaemun_test)

# 서초구 다중회귀
model_Seochu = LinearRegression()
X_Seochu = X[df['자치구명'] == '서초구'].drop(columns=['자치구명'])
Y_Seochu = Y[df['자치구명'] == '서초구']
X_Seochu_train, X_Seochu_test, Y_Seochu_train, Y_Seochu_test = train_test_split(X_Seochu, Y_Seochu, test_size=0.2)
print('서초구 Model Learning Start')
model_Seochu.fit(X_Seochu_train, Y_Seochu_train)
print('서초구 Model Learning End')
print('서초구 Train Score :', model_Seochu.score(X_Seochu_train, Y_Seochu_train))
print('서초구 Test Score :', model_Seochu.score(X_Seochu_test, Y_Seochu_test))
train_score_sum += model_Seochu.score(X_Seochu_train, Y_Seochu_train)
test_score_sum += model_Seochu.score(X_Seochu_test, Y_Seochu_test)

# 성동구 다중회귀
model_Seongdong = LinearRegression()
X_Seongdong = X[df['자치구명'] == '성동구'].drop(columns=['자치구명'])
Y_Seongdong = Y[df['자치구명'] == '성동구']
X_Seongdong_train, X_Seongdong_test, Y_Seongdong_train, Y_Seongdong_test = train_test_split(X_Seongdong, Y_Seongdong, test_size=0.2)
print('성동구 Model Learning Start')
model_Seongdong.fit(X_Seongdong_train, Y_Seongdong_train)
print('성동구 Model Learning End')
print('성동구 Train Score :', model_Seongdong.score(X_Seongdong_train, Y_Seongdong_train))
print('성동구 Test Score :', model_Seongdong.score(X_Seongdong_test, Y_Seongdong_test))
train_score_sum += model_Seongdong.score(X_Seongdong_train, Y_Seongdong_train)
test_score_sum += model_Seongdong.score(X_Seongdong_test, Y_Seongdong_test)

# 성북구 다중회귀
model_Seongbuk = LinearRegression()
X_Seongbuk = X[df['자치구명'] == '성북구'].drop(columns=['자치구명'])
Y_Seongbuk = Y[df['자치구명'] == '성북구']
X_Seongbuk_train, X_Seongbuk_test, Y_Seongbuk_train, Y_Seongbuk_test = train_test_split(X_Seongbuk, Y_Seongbuk, test_size=0.2)
print('성북구 Model Learning Start')
model_Seongbuk.fit(X_Seongbuk_train, Y_Seongbuk_train)
print('성북구 Model Learning End')
print('성북구 Train Score :', model_Seongbuk.score(X_Seongbuk_train, Y_Seongbuk_train))
print('성북구 Test Score :', model_Seongbuk.score(X_Seongbuk_test, Y_Seongbuk_test))
train_score_sum += model_Seongbuk.score(X_Seongbuk_train, Y_Seongbuk_train)
test_score_sum += model_Seongbuk.score(X_Seongbuk_test, Y_Seongbuk_test)

# 송파구 다중회귀
model_Songpa = LinearRegression()
X_Songpa = X[df['자치구명'] == '송파구'].drop(columns=['자치구명'])
Y_Songpa = Y[df['자치구명'] == '송파구']
X_Songpa_train, X_Songpa_test, Y_Songpa_train, Y_Songpa_test = train_test_split(X_Songpa, Y_Songpa, test_size=0.2)
print('송파구 Model Learning Start')
model_Songpa.fit(X_Songpa_train, Y_Songpa_train)
print('송파구 Model Learning End')
print('송파구 Train Score :', model_Songpa.score(X_Songpa_train, Y_Songpa_train))
print('송파구 Test Score :', model_Songpa.score(X_Songpa_test, Y_Songpa_test))
train_score_sum += model_Songpa.score(X_Songpa_train, Y_Songpa_train)
test_score_sum += model_Songpa.score(X_Songpa_test, Y_Songpa_test)

# 양천구 다중회귀
model_Yangcheon = LinearRegression()
X_Yangcheon = X[df['자치구명'] == '양천구'].drop(columns=['자치구명'])
Y_Yangcheon = Y[df['자치구명'] == '양천구']
X_Yangcheon_train, X_Yangcheon_test, Y_Yangcheon_train, Y_Yangcheon_test = train_test_split(X_Yangcheon, Y_Yangcheon, test_size=0.2)
print('양천구 Model Learning Start')
model_Yangcheon.fit(X_Yangcheon_train, Y_Yangcheon_train)
print('양천구 Model Learning End')
print('양천구 Train Score :', model_Yangcheon.score(X_Yangcheon_train, Y_Yangcheon_train))
print('양천구 Test Score :', model_Yangcheon.score(X_Yangcheon_test, Y_Yangcheon_test))
train_score_sum += model_Yangcheon.score(X_Yangcheon_train, Y_Yangcheon_train)
test_score_sum += model_Yangcheon.score(X_Yangcheon_test, Y_Yangcheon_test)

# 영등포구 다중회귀
model_Yeongdeungpo = LinearRegression()
X_Yeongdeungpo = X[df['자치구명'] == '영등포구'].drop(columns=['자치구명'])
Y_Yeongdeungpo = Y[df['자치구명'] == '영등포구']
X_Yeongdeungpo_train, X_Yeongdeungpo_test, Y_Yeongdeungpo_train, Y_Yeongdeungpo_test = train_test_split(X_Yeongdeungpo, Y_Yeongdeungpo, test_size=0.2)
print('영등포구 Model Learning Start')
model_Yeongdeungpo.fit(X_Yeongdeungpo_train, Y_Yeongdeungpo_train)
print('영등포구 Model Learning End')
print('영등포구 Train Score :', model_Yeongdeungpo.score(X_Yeongdeungpo_train, Y_Yeongdeungpo_train))
print('영등포구 Test Score :', model_Yeongdeungpo.score(X_Yeongdeungpo_test, Y_Yeongdeungpo_test))
train_score_sum += model_Yeongdeungpo.score(X_Yeongdeungpo_train, Y_Yeongdeungpo_train)
test_score_sum += model_Yeongdeungpo.score(X_Yeongdeungpo_test, Y_Yeongdeungpo_test)

# 용산구 다중회귀
model_Yongsan = LinearRegression()
X_Yongsan = X[df['자치구명'] == '용산구'].drop(columns=['자치구명'])
Y_Yongsan = Y[df['자치구명'] == '용산구']
X_Yongsan_train, X_Yongsan_test, Y_Yongsan_train, Y_Yongsan_test = train_test_split(X_Yongsan, Y_Yongsan, test_size=0.2)
print('용산구 Model Learning Start')
model_Yongsan.fit(X_Yongsan_train, Y_Yongsan_train)
print('용산구 Model Learning End')
print('용산구 Train Score :', model_Yongsan.score(X_Yongsan_train, Y_Yongsan_train))
print('용산구 Test Score :', model_Yongsan.score(X_Yongsan_test, Y_Yongsan_test))
train_score_sum += model_Yongsan.score(X_Yongsan_train, Y_Yongsan_train)
test_score_sum += model_Yongsan.score(X_Yongsan_test, Y_Yongsan_test)

# 은평구 다중회귀
model_Eunpyeong = LinearRegression()
X_Eunpyeong = X[df['자치구명'] == '은평구'].drop(columns=['자치구명'])
Y_Eunpyeong = Y[df['자치구명'] == '은평구']
X_Eunpyeong_train, X_Eunpyeong_test, Y_Eunpyeong_train, Y_Eunpyeong_test = train_test_split(X_Eunpyeong, Y_Eunpyeong, test_size=0.2)
print('은평구 Model Learning Start')
model_Eunpyeong.fit(X_Eunpyeong_train, Y_Eunpyeong_train)
print('은평구 Model Learning End')
print('은평구 Train Score :', model_Eunpyeong.score(X_Eunpyeong_train, Y_Eunpyeong_train))
print('은평구 Test Score :', model_Eunpyeong.score(X_Eunpyeong_test, Y_Eunpyeong_test))
train_score_sum += model_Eunpyeong.score(X_Eunpyeong_train, Y_Eunpyeong_train)
test_score_sum += model_Eunpyeong.score(X_Eunpyeong_test, Y_Eunpyeong_test)

# 종로구 다중회귀
model_Jongno = LinearRegression()
X_Jongno = X[df['자치구명'] == '종로구'].drop(columns=['자치구명'])
Y_Jongno = Y[df['자치구명'] == '종로구']
X_Jongno_train, X_Jongno_test, Y_Jongno_train, Y_Jongno_test = train_test_split(X_Jongno, Y_Jongno, test_size=0.2)
print('종로구 Model Learning Start')
model_Jongno.fit(X_Jongno_train, Y_Jongno_train)
print('종로구 Model Learning End')
print('종로구 Train Score :', model_Jongno.score(X_Jongno_train, Y_Jongno_train))
print('종로구 Test Score :', model_Jongno.score(X_Jongno_test, Y_Jongno_test))
train_score_sum += model_Jongno.score(X_Jongno_train, Y_Jongno_train)
test_score_sum += model_Jongno.score(X_Jongno_test, Y_Jongno_test)

# 중구 다중회귀
model_Jung = LinearRegression()
X_Jung = X[df['자치구명'] == '중구'].drop(columns=['자치구명'])
Y_Jung = Y[df['자치구명'] == '중구']
X_Jung_train, X_Jung_test, Y_Jung_train, Y_Jung_test = train_test_split(X_Jung, Y_Jung, test_size=0.2)
print('중구 Model Learning Start')
model_Jung.fit(X_Jung_train, Y_Jung_train)
print('중구 Model Learning End')
print('중구 Train Score :', model_Jung.score(X_Jung_train, Y_Jung_train))
print('중구 Test Score :', model_Jung.score(X_Jung_test, Y_Jung_test))
train_score_sum += model_Jung.score(X_Jung_train, Y_Jung_train)
test_score_sum += model_Jung.score(X_Jung_test, Y_Jung_test)

# 중랑구 다중회귀
model_Jungnang = LinearRegression()
X_Jungnang = X[df['자치구명'] == '중랑구'].drop(columns=['자치구명'])
Y_Jungnang = Y[df['자치구명'] == '중랑구']
X_Jungnang_train, X_Jungnang_test, Y_Jungnang_train, Y_Jungnang_test = train_test_split(X_Jungnang, Y_Jungnang, test_size=0.2)
print('중랑구 Model Learning Start')
model_Jungnang.fit(X_Jungnang_train, Y_Jungnang_train)
print('중랑구 Model Learning End')
print('중랑구 Train Score :', model_Jungnang.score(X_Jungnang_train, Y_Jungnang_train))
print('중랑구 Test Score :', model_Jungnang.score(X_Jungnang_test, Y_Jungnang_test))
train_score_sum += model_Jungnang.score(X_Jungnang_train, Y_Jungnang_train)
test_score_sum += model_Jungnang.score(X_Jungnang_test, Y_Jungnang_test)

print('Average Train Score :', train_score_sum / 25)
print('Average Test Score :', test_score_sum / 25)
# Average Train Score : 0.4775744823960981 / Average Test Score : 0.4756067276979516
# 과소적합 가능성 존재.

# nameJachigu = ['강남구', '강동구', '강북구', '강서구', '관악구', '광진구', '구로구', '금천구', '노원구', '도봉구', '동대문구', '동작구', '마포구', '서대문구', '서초구', '성동구', '성북구', '송파구', '양천구', '영등포구', '용산구', '은평구', '종로구', '중구', '중랑구']
# print(len(nameJachigu))
# 그래프로 나타내기
# 입력값을 통한 예측
# Column : ['자치구명', '물건금액(만원)', '건물면적(㎡)', '층']
print('----------------------------------------')
print('<서울 부동산 가격 예측 프로그램>')
print('이 프로그램은 무료로 가격을 예측해 줍니다!')
print('자치구 : 강남구, 강동구, 강북구, 강서구, 관악구, 광진구, 구로구, 금천구, 노원구, 도봉구, 동대문구, 동작구, 마포구, 서대문구, 서초구, 성동구, 성북구, 송파구, 양천구, 영등포구, 용산구, 은평구, 종로구, 중구, 중랑구')
print('----------------------------------------')
nameJachigu = ['강남구', '강동구', '강북구', '강서구', '관악구', '광진구', '구로구', '금천구', '노원구', '도봉구', '동대문구', '동작구', '마포구', '서대문구', '서초구', '성동구', '성북구', '송파구', '양천구', '영등포구', '용산구', '은평구', '종로구', '중구', '중랑구']
while True:
    try:
        while True:
            Jachigu = input('자치구명 : ')
            if Jachigu in nameJachigu:
                break
            else:
                print('올바르지 못한 자치구입니다!')

        Area = float(input('건물면적(㎡) : '))
        Floor = int(input('층 : '))
        arr = [[Area, Floor]]
        price_predicted = 0
        df_input = pd.DataFrame(arr, columns=['건물면적(㎡)', '층']).to_numpy()
        if Jachigu == '강남구':
            price_predicted = int(model_Gangnam.predict(df_input)[0]) * 10000
        elif Jachigu == '강동구':
            price_predicted = int(model_Gangdong.predict(df_input)[0]) * 10000
        elif Jachigu == '강북구':
            price_predicted = int(model_Gangbuk.predict(df_input)[0]) * 10000
        elif Jachigu == '강서구':
            price_predicted = int(model_Gangseo.predict(df_input)[0]) * 10000
        elif Jachigu == '관악구':
            price_predicted = int(model_Gwanak.predict(df_input)[0]) * 10000
        elif Jachigu == '광진구':
            price_predicted = int(model_Gwangjin.predict(df_input)[0]) * 10000
        elif Jachigu == '구로구':
            price_predicted = int(model_Guro.predict(df_input)[0]) * 10000
        elif Jachigu == '금천구':
            price_predicted = int(model_Geumcheon.predict(df_input)[0]) * 10000
        elif Jachigu == '노원구':
            price_predicted = int(model_Nowon.predict(df_input)[0]) * 10000
        elif Jachigu == '도봉구':
            price_predicted = int(model_Dobong.predict(df_input)[0]) * 10000
        elif Jachigu == '동대문구':
            price_predicted = int(model_Dongdaemun.predict(df_input)[0]) * 10000
        elif Jachigu == '동작구':
            price_predicted = int(model_Dongjak.predict(df_input)[0]) * 10000
        elif Jachigu == '마포구':
            price_predicted = int(model_Mapo.predict(df_input)[0]) * 10000
        elif Jachigu == '서대문구':
            price_predicted = int(model_Seodaemun.predict(df_input)[0]) * 10000
        elif Jachigu == '서초구':
            price_predicted = int(model_Seochu.predict(df_input)[0]) * 10000
        elif Jachigu == '성동구':
            price_predicted = int(model_Seongdong.predict(df_input)[0]) * 10000
        elif Jachigu == '성북구':
            price_predicted = int(model_Seongbuk.predict(df_input)[0]) * 10000
        elif Jachigu == '송파구':
            price_predicted = int(model_Songpa.predict(df_input)[0]) * 10000
        elif Jachigu == '양천구':
            price_predicted = int(model_Yangcheon.predict(df_input)[0]) * 10000
        elif Jachigu == '영등포구':
            price_predicted = int(model_Yeongdeungpo.predict(df_input)[0]) * 10000
        elif Jachigu == '용산구':
            price_predicted = int(model_Yongsan.predict(df_input)[0]) * 10000
        elif Jachigu == '은평구':
            price_predicted = int(model_Eunpyeong.predict(df_input)[0]) * 10000
        elif Jachigu == '종로구':
            price_predicted = int(model_Jongno.predict(df_input)[0]) * 10000
        elif Jachigu == '중구':
            price_predicted = int(model_Jung.predict(df_input)[0]) * 10000
        elif Jachigu == '중랑구':
            price_predicted = int(model_Jungnang.predict(df_input)[0]) * 10000
        print('예측 가격 :', format(price_predicted, ',d'), '원')
        print('----------------------------------------')
    except UserWarning:
        continue

    except KeyboardInterrupt:
        print('\n프로그램을 종료합니다.')
        break