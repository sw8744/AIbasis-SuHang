from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

model_main = pickle.load(open('main.pkl', 'rb'))
model_tree = pickle.load(open('main_tree.pkl', 'rb'))
model_KNN = pickle.load(open('main_KNN.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def main():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def home():
    if request.method == 'GET' or request.method == 'POST':
        Jachigu = request.form['Jachigu']
        Area = float(request.form['Area'])
        Floor = int(request.form['Floor'])
        nameJachigu = ['강남구', '강동구', '강북구', '강서구', '관악구', '광진구', '구로구', '금천구', '노원구', '도봉구', '동대문구', '동작구', '마포구',
        '서대문구', '서초구', '성동구', '성북구', '송파구', '양천구', '영등포구', '용산구', '은평구', '종로구', '중구', '중랑구']
        Jachigu_final = nameJachigu.index(Jachigu)
        arr = np.array([[Jachigu_final, Area, Floor]])
        df_input = pd.DataFrame(arr, columns=['자치구명', '건물면적(㎡)', '층'])
        pred_main = int(model_main.predict(df_input)) * 10000
        pred_main = format(pred_main, ',d')
        pred_tree = int(model_tree.predict(df_input)) * 10000
        pred_tree = format(pred_tree, ',d')
        pred_KNN = int(model_KNN.predict(df_input)) * 10000
        pred_KNN = format(pred_KNN, ',d')
        return render_template('predict.html', pred_main=pred_main, pred_tree=pred_tree, pred_KNN=pred_KNN)

if __name__ == '__main__':
    app.run(debug=True)