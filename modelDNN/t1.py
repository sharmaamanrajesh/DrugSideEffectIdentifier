from tensorflow.keras.models import model_from_json
import joblib
import numpy as np

# with open("model_pickle",'rb') as f:
    # mod = pickle.load(f)

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
mod = model_from_json(loaded_model_json)
# load weights into new model
mod.load_weights("model.h5")
print("Loaded model from disk")



# mod = joblib.load("joblib_model.pkl")
    

Test_Drug = "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 1 0 0 1 0 1 0 1 0 1 0 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 1 0 0 1 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 1 0 1 0 0 0 0 1 0 1 1 0 0 1 1 1 1 0 1 1 1 1 1 0 0 1 0 0"

TestDrug = Test_Drug.split()

TestDrug1 = []

for i in TestDrug:
    TestDrug1.append(float(i))

Test_Drug = np.array(TestDrug1)
Test_Drug = np.arange(167).reshape(1,167)

resultArr = mod.predict(Test_Drug)
counter = 0
for i in resultArr[0]:
    if i:
        counter += 1

print(f"hello {counter}")