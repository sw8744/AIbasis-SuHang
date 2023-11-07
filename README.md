# 다중회귀 모델을 이용한 서울 지역 주택 가격 예측 프로그램
## 1. 개요
이 프로그램은 Scikit-Learn 라이브러리에서 제공하는 다중회귀 모델과 서울열린데이터광장에서 제공하는 [서울시 부동산 실거래가 정보](https://data.seoul.go.kr/dataList/OA-21275/S/1/datasetView.do)를 이용하여 서울 지역의 주택 가격을 예측하는 프로그램이다.

접속 주소 : http://aibasis.kro.kr

## 2. 사용법
### 1. 기본 설정
프로젝트 폴더 안에 Dataset 이라는 폴더를 만든 후, 여기에 [Seoul_House_Price.csv](https://drive.google.com/file/d/1mCtFQX-__AS8-88yXl4WCxcuIhsSeiYh/view?usp=sharing)를 다운로드 받아 폴더 안에 붙여넣고 실행해야 한다.

### 2. 라이브러리 설정
```
pip install scikit-learn
pip install pandas
pip install matplotlib
```
위의 명령어를 한 줄 씩 CMD에 입력하여 라이브러리를 설치한다.

### 3. 결과
> Train Score : 0.42515822592183095
> 
> Test Score : 0.4257265982893499

이런 식으로 Train Score와 Test Score가 비슷하게 나오므로 과소적합이라 볼 수 있다.

### 4. 개선
#### 1. 지역구별로 모델 쪼개기
지역구별로 부동산 가격의 편차가 심할 것으로 예측되어 지역별로 모델을 나누었다.
> Average Train Score : 0.4775744823960981
> 
> Average Test Score : 0.4756067276979516

아주 소폭 상승한 것을 알 수 있다. 이를 통해 지역별로 모델을 나누는 것이 효과적임을 알 수 있다.

#### 2. 다른 모델 사용하기 - KNN
##### KNN이란?
알고리즘의 하나로, 특정 점에서 가장 가까운 점들의 개수를 따지고, 가장 많이 나온 값으로 분류하는 알고리즘


> Train Score : 0.7190557420613956
> 
> Test Score : 0.6766586516808314

점수가 큰 폭으로 상승한 것을 알 수 있다. 이를 통해 KNN 알고리즘이 효과적이라는 것을 알 수 있다.

#### 3. 다른 모델 사용하기 - Decision Tree
##### Decision Tree란?
알고리즘의 하나로, 특정 조건에 따라 분류하는 알고리즘

> Train Score : 0.8199007481525984
> 
> Test Score : 0.7042106067856249

점수가 기존 다중회귀 때보다 거의 2배가량 상승한 것으로 미루어 보아 Decision Tree 알고리즘이 효과적이라는 것을 알 수 있다.