import joblib
import pandas as pd

model = joblib.load('car_price_prediction.pkl')

sample = pd.DataFrame({
    'Present_Price':[5.0],
    'Kms_Driven':[10000],
    'Owner':[0],
    'Car_Age':[3],
    'Fuel_Type_Diesel':[0],
    'Fuel_Type_Petrol':[1],
    'Seller_Type_Individual':[0],
    'Transmission_Manual':[1]
})

print('Loaded model:', type(model))
print('Prediction:', model.predict(sample))
