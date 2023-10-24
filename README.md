# 다중회귀 모델을 이용한 서울 지역 주택 가격 예측 프로그램
## 1. 개요
이 프로그램은 Scikit-Learn 라이브러리에서 제공하는 다중회귀 모델과 서울열린데이터광장에서 제공하는 [서울시 부동산 실거래가 정보](https://data.seoul.go.kr/dataList/OA-21275/S/1/datasetView.do)를 이용하여 서울 지역의 주택 가격을 예측하는 프로그램이다.

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
> Test Score : 0.4257265982893499

이런 식으로 Train Score와 Test Score가 비슷하게 나오므로 과소적합이라 볼 수 있다.

다중회귀보다 KNN이 더 잘 나오긴 한다. 72% 정도. KNN 쓰면 안되나? ㅠ