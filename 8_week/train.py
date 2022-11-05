import numpy as np

import bentoml

from bentoml.io import JSON

model_ref = bentoml.xgboost.get("maintenance_predict_model:asw4gns4q2vxgjv5")
dv = model_ref.custom_objects["dictVectorizer"]

model_runner = model_ref.to_runner()

svc = bentoml.Service("maintenance_predict_classifier", runners=[model_runner])

@svc.api(input=JSON(), output=JSON())
def classify(machine_data):
    vector = dv.transform(machine_data)
    prediction = model_runner.predict.run(vector)
    print(prediction)
    result = prediction[0]
    if result > 0.75:
        return {"status": "DO MAINTENANCE"}
    elif result > 0.5:
        return {"status": "PROPENSE TO FAIL"}
    else:
        return {"status": "MACHINE OK"}
