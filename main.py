# 데이터 불러오기
import pandas as pd
df = pd.read_csv('Dataset/Seoul_House_Price.csv', encoding='cp949', low_memory=False)
# 필요 없는 칼럼 삭제
df.drop(columns=['접수연도', '자치구코드', '법정동코드', '법정동명', '지번구분', '지번구분명', '본번', '부번', '건물명', '건물용도', '계약일', '토지면적(㎡)', '권리구분', '취소일', '건축년도', '신고구분', '신고한 개업공인중개사 시군구명'], inplace=True)
# print(df)

# 결과값과 입력값 분리
Y = df['물건금액(만원)'].to_numpy()
X = df.drop(columns=['물건금액(만원)']).to_numpy()

# 훈련 데이터와 테스트 데이터 분리
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
# print(X_train)
print('Data Prepared')

# 문자(자치구명, 건물용도)를 숫자로 변경
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(X_train[:, 0])
X_train[:, 0] = le.transform(X_train[:, 0])
X_test[:, 0] = le.transform(X_test[:, 0])
# print(X_train)

# 다중회귀 알고리즘 사용
# nameJachigu = ['강남구', '강동구', '강북구', '강서구', '관악구', '광진구', '구로구', '금천구', '노원구', '도봉구', '동대문구', '동작구', '마포구', '서대문구', '서초구', '성동구', '성북구', '송파구', '양천구', '영등포구', '용산구', '은평구', '종로구', '중구', '중랑구']
# print(len(nameJachigu))
from sklearn.linear_model import LinearRegression
model = LinearRegression()
print('Model Learning Start')
model.fit(X_train, Y_train)
print('Model Learning End')
# print(model.predict(X_test))
print('Train Score :', model.score(X_train, Y_train))
print('Test Score :', model.score(X_test, Y_test))
# Train Score : 0.42515822592183095, Test Score : 0.4257265982893499
# Train Data가 Test Data보다 더 작으므로 과소적합됨을 알 수 있음.

# 그래프에 나타내기 위한 차원 축소
from sklearn.decomposition import PCA
pca = PCA(n_components=1) # 1차원으로 축소
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print('Finish PCA')
# print(X_train_pca)

# 그래프로 나타내기
import matplotlib.pyplot as plt
plt.scatter(X_train_pca, Y_train, color='red')
plt.xlabel('PCA')
plt.ylabel('Price')
plt.show()

# 입력값을 통한 예측
# Column : ['자치구명', '물건금액(만원)', '건물면적(㎡)', '층']
print('----------------------------------------')
print('<서울 부동산 가격 예측 프로그램>')
print('이 프로그램은 무료로 가격을 예측해 줍니다!')
print('----------------------------------------')
nameJachigu = ['강남구', '강동구', '강북구', '강서구', '관악구', '광진구', '구로구', '금천구', '노원구', '도봉구', '동대문구', '동작구', '마포구', '서대문구', '서초구', '성동구', '성북구', '송파구', '양천구', '영등포구', '용산구', '은평구', '종로구', '중구', '중랑구']
while True:
    Jachigu = input('자치구명 : ')
    if Jachigu in nameJachigu:
        break
    else:
        print('올바르지 못한 자치구입니다!')
Area = float(input('건물면적(㎡) : '))
Floor = int(input('층 : '))
arr = [[Jachigu, Area, Floor]]
df_input = pd.DataFrame(arr, columns=['자치구명', '건물면적(㎡)', '층']).to_numpy()
df_input[:, 0] = le.transform(df_input[:, 0])
price_predicted = int(model.predict(df_input)[0]) * 10000
print('예측 가격 :', format(price_predicted, ',d'), '원')