import numpy as np

import bentoml

from bentoml.io import JSON

model_ref = bentoml.sklearn.get("flight_price_prediction:wqqm6cd7xcjtzzc6")
dv = model_ref.custom_objects["dictVectorizer"]

model_runner = model_ref.to_runner()

svc = bentoml.Service("flight_price_prediction", runners=[model_runner])

@svc.api(input=JSON(), output=JSON())
def predict(price_flight):
    vector = dv.transform(price_flight)
    prediction = model_runner.predict.run(vector)
    print(prediction)
    return round(prediction[0], 2)
