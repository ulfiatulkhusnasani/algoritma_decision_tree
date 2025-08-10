from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

app = FastAPI()

# Fungsi untuk menentukan label 'kinerja'
def tentukan_kinerja(kehadiran, produktivitas):
    if kehadiran >= 240 and produktivitas >= 80:
        return 'Baik'
    elif kehadiran >= 220 and produktivitas >= 65:
        return 'Cukup'
    else:
        return 'Kurang'

# Membuat data 100 baris
data = {
    'kehadiran': [],
    'target_kehadiran': [],
    'produktivitas': [],
    'target_produktivitas': [],
    'kinerja': []
}

for _ in range(100):
    kehadiran = random.randint(190, 260)
    produktivitas = random.randint(45, 100)
    target_kehadiran = 260
    target_produktivitas = 80
    kinerja = tentukan_kinerja(kehadiran, produktivitas)

    data['kehadiran'].append(kehadiran)
    data['target_kehadiran'].append(target_kehadiran)
    data['produktivitas'].append(produktivitas)
    data['target_produktivitas'].append(target_produktivitas)
    data['kinerja'].append(kinerja)

# Buat DataFrame
df = pd.DataFrame(data)

# Pisahkan fitur dan target
X = df[['kehadiran', 'target_kehadiran', 'produktivitas', 'target_produktivitas']]
y = df['kinerja']

# Bagi data untuk training dan testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Latih model Decision Tree
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Evaluasi akurasi
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("=== Evaluasi Model ===")
print(f"Akurasi: {accuracy:.2%}")
print("\nLaporan Klasifikasi:")
print(classification_report(y_test, y_pred))

# Schema input untuk prediksi
class KaryawanInput(BaseModel):
    kehadiran: int
    target_kehadiran: int
    produktivitas: int
    target_produktivitas: int
    bonus_jabatan: int

# Endpoint prediksi
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
        "bonus": bonus,
        "akurasi_model": f"{accuracy:.2%}"
    }
