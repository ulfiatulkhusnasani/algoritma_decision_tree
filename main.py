from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

app = FastAPI()

# Data dummy dan pelatihan model
data = {
    'kehadiran': [250, 240, 230, 210, 220, 230, 200, 240, 250, 210],
    'target_kehadiran': [260] * 10,
    'produktivitas': [85, 80, 70, 60, 90, 65, 50, 70, 95, 60],
    'target_produktivitas': [80] * 10,
    'kinerja': ['Baik', 'Baik', 'Cukup', 'Kurang', 'Baik', 'Cukup', 'Kurang', 'Cukup', 'Baik', 'Kurang']
}

df = pd.DataFrame(data)
X = df[['kehadiran', 'target_kehadiran', 'produktivitas', 'target_produktivitas']]
y = df['kinerja']

model = DecisionTreeClassifier()
model.fit(X, y)

class KaryawanInput(BaseModel):
    kehadiran: int
    target_kehadiran: int
    produktivitas: int
    target_produktivitas: int
    bonus_jabatan: int

@app.post("/prediksi/")
def prediksi_kinerja(data: KaryawanInput):

    data_predict = {
        'kehadiran': data.kehadiran,
        'target_kehadiran': data.target_kehadiran,
        'produktivitas': data.produktivitas,
        'target_produktivitas': data.target_produktivitas
    }

    input_df = pd.DataFrame([data_predict])
    pred_kinerja = model.predict(input_df)[0]
    selisih = data.produktivitas - data.target_produktivitas
    bonus = selisih * data.bonus_jabatan if selisih > 0 and pred_kinerja == 'Baik' else 0

    return {
        "kinerja": pred_kinerja,
        "bonus": bonus
    }
