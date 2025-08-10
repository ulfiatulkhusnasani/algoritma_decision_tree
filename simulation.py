import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Data 100 dummy
data = {
    'kehadiran': [
        250, 240, 230, 210, 220, 230, 200, 240, 250, 210,
        260, 255, 245, 235, 225, 215, 205, 195, 250, 240,
        260, 250, 240, 230, 220, 210, 200, 190, 255, 245,
        260, 250, 240, 230, 220, 210, 200, 195, 255, 245,
        260, 250, 240, 230, 220, 210, 200, 195, 250, 240,
        260, 255, 245, 235, 225, 215, 205, 195, 250, 240,
        260, 250, 240, 230, 220, 210, 200, 190, 255, 245,
        260, 250, 240, 230, 220, 210, 200, 195, 255, 245,
        260, 250, 240, 230, 220, 210, 200, 195, 250, 240,
        260, 255, 245, 235, 225, 215, 205, 195, 250, 240
    ],
    'target_kehadiran': [260] * 100,
    'produktivitas': [
        85, 80, 70, 60, 90, 65, 50, 70, 95, 60,
        88, 82, 78, 68, 66, 58, 48, 46, 85, 80,
        92, 86, 72, 64, 60, 55, 48, 45, 88, 70,
        90, 84, 74, 66, 60, 55, 50, 46, 86, 78,
        93, 88, 80, 72, 68, 60, 50, 48, 85, 78,
        95, 90, 82, 74, 70, 64, 58, 50, 88, 82,
        92, 86, 76, 68, 64, 58, 50, 48, 90, 78,
        94, 88, 80, 72, 68, 62, 55, 50, 88, 82,
        96, 90, 82, 74, 70, 65, 58, 50, 85, 78,
        95, 90, 84, 76, 70, 66, 60, 55, 88, 82
    ],
    'target_produktivitas': [80] * 100,
    'kinerja': [
        'Baik', 'Baik', 'Cukup', 'Kurang', 'Baik', 'Cukup', 'Kurang', 'Cukup', 'Baik', 'Kurang',
        'Baik', 'Baik', 'Cukup', 'Cukup', 'Cukup', 'Kurang', 'Kurang', 'Kurang', 'Baik', 'Baik',
        'Baik', 'Baik', 'Cukup', 'Kurang', 'Kurang', 'Kurang', 'Kurang', 'Kurang', 'Baik', 'Cukup',
        'Baik', 'Baik', 'Cukup', 'Cukup', 'Kurang', 'Kurang', 'Kurang', 'Kurang', 'Baik', 'Cukup',
        'Baik', 'Baik', 'Baik', 'Cukup', 'Cukup', 'Kurang', 'Kurang', 'Kurang', 'Baik', 'Cukup',
        'Baik', 'Baik', 'Baik', 'Cukup', 'Cukup', 'Kurang', 'Kurang', 'Kurang', 'Baik', 'Baik',
        'Baik', 'Baik', 'Cukup', 'Cukup', 'Kurang', 'Kurang', 'Kurang', 'Kurang', 'Baik', 'Cukup',
        'Baik', 'Baik', 'Baik', 'Cukup', 'Cukup', 'Kurang', 'Kurang', 'Kurang', 'Baik', 'Baik',
        'Baik', 'Baik', 'Baik', 'Cukup', 'Cukup', 'Kurang', 'Kurang', 'Kurang', 'Baik', 'Cukup',
        'Baik', 'Baik', 'Baik', 'Cukup', 'Cukup', 'Kurang', 'Kurang', 'Kurang', 'Baik', 'Baik'
    ]
}

# Buat DataFrame
df = pd.DataFrame(data)

# Pisahkan fitur dan target
X = df[['kehadiran', 'target_kehadiran', 'produktivitas', 'target_produktivitas']]
y = df['kinerja']

# Split train/test
# test size ku buat jadi 0.2 karena agar digunakan sebagian besar untuk training dan sebgaian kecil untuk testing data yang baru 80% training 20% untuk data testing 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Latih model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Evaluasi
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred, labels=['Baik', 'Cukup', 'Kurang'])
cm_df = pd.DataFrame(cm,
                     index=['Actual Baik', 'Actual Cukup', 'Actual Kurang'],
                     columns=['Pred Baik', 'Pred Cukup', 'Pred Kurang'])

totActualBaik = totActualCukup = totActualKurang = 0
totPredictBaik = totPredictCukup = totPredictKurang = 0

for actual, pred in zip(y_test, y_pred):
    match actual:
        case "Baik":
            totActualBaik += 1
        case "Cukup":
            totActualCukup += 1
        case _:
            totActualKurang += 1
    match pred:
        case "Baik":
            totPredictBaik += 1
        case "Cukup":
            totPredictCukup += 1
        case _:
            totPredictKurang += 1

# Data total actual & predict
totals_df = pd.DataFrame({
    "Kategori": ["Baik", "Cukup", "Kurang"],
    "Total Actual": [totActualBaik, totActualCukup, totActualKurang],
    "Total Predict": [totPredictBaik, totPredictCukup, totPredictKurang]
})

# Laporan klasifikasi dalam bentuk DataFrame
report_dict = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()

# ====== EXPORT KE EXCEL ======
with pd.ExcelWriter("evaluasi_model.xlsx") as writer:
    totals_df.to_excel(writer, sheet_name="Totals", index=False)
    cm_df.to_excel(writer, sheet_name="Confusion Matrix")
    report_df.to_excel(writer, sheet_name="Classification Report")

print("Data evaluasi berhasil disimpan ke evaluasi_model.xlsx")


# jadi dari 20 data testing kita dapat akurasi 100% karena dari data yang kita dapat di actual itu sama dengan yang ada di data prediksi
# untuk actual baik terdapat 9 data benar 
# untuk actual cukup terdapat 8 data benar
# untuk actual kurang terdapat 3 data benar

# untuk prediction baik terdapat 9 data benar 
# untuk prediction cukup terdapat 8 data benar
# untuk prediction kurang terdapat 3 data benar

