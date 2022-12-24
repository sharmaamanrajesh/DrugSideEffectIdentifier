from django.shortcuts import render,redirect
from django.http import HttpResponse
import pandas as pd
from tensorflow.keras.models import model_from_json
import joblib
import numpy as np
import csv
import pickle
# Create your views here.

# myDataSet = {
# 'SE_1':[1],
# 'SE_2':[0],
# 'SE_3':[0],
# 'SE_4':[1],
# 'SE_5':[1],
# }
# # myPredictions = [1,0,0,1,1]

# myDataFrame = pd.DataFrame(myDataSet)
# # myDataFrame

# # print(myDataFrame)

df = pd.read_csv("static/SMILES with ADR.csv")
y = df.iloc[:,169:]
sideEffectLabelsColumns = y.columns.tolist()
# print(sideEffectLabelsColumns)

val = ''

def home(request):
    val = ""
    return render(request,'project.html',
    {'PredictedOutcome':val})

def predictIt(request):

    if request.method == 'POST':

        global val
        result = ""
        # global resultArr

        val = request.POST['InputBar']

        TestDrug = val.split()

        TestDrug1 = []

        for i in TestDrug:
            TestDrug1.append(float(i))

        Test_Drug = np.array(TestDrug1)

        Test_Drug = np.arange(167).reshape(1,167)


        json_file = open('static/model.json')
        loaded_model_json = json_file.read()
        json_file.close()
        mod = model_from_json(loaded_model_json)
        # load weights into new model
        mod.load_weights("static/model.h5")
        # print("Loaded model from disk")

        # with open("modelDNN/model_pickle.zip",'rb') as f:
        #     mod = pickle.load(f)

        resultTemp = mod.predict(Test_Drug)
        resultArr = np.where(resultTemp >= 0.5, 1, 0)
        
        for i in range(0,6123):
            if resultArr[0][i] == 1:
                # print(sideEffectLabelsColumns[i])
                result = result + sideEffectLabelsColumns[i] + "\n"
        
        return render(request,'project.html',{'PredictedOutcome':result,'Initial_167_Code':val})


        # myNums = val.split()

        # print(myNums)
        # myNewNums = []
        # for i in myNums:
        #     myNewNums.append(int(i))
        
        # myNewNums.sort();
        # print(myNewNums)
        # result = ""
        # for i in myNewNums:
        #     result = result + str(i)+" "

        # print(result)
        # col = myDataFrame.head()
        # # print(col)
        # columns = list(col)
        # for i in columns:
        #     if(col[i][0]):
        #         print(i,col[i][0])
        #         result = result + i + '\n'



    if request.method == 'GET':
        return redirect('home')

    return render(request,'test1.html')