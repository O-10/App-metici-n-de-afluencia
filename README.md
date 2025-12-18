import cv2
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from ultralytics import YOLO
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import os

# ============================================
# 1. CONFIGURACIÓN INICIAL
# ============================================

print("Cargando modelo YOLOv8 para detección de personas...")
model = YOLO("yolov8s.pt")  # Descarga automática la primera vez. Usa yolov8m.pt para más precisión
PERSON_CLASS_ID = 0

print("Cargando modelo de texto para clustering...")
text_model = SentenceTransformer("all-MiniLM-L6-v2")

# Preguntar área visible
try:
    area_visible = float(input("\n¿Cuántos metros cuadrados cubre aproximadamente el área visible de la cámara? "
                                "\n(Ej: si está a 4-5m de altura → ingresa 30): "))
    if area_visible <= 0:
        raise ValueError
except:
    print("Valor inválido. Usando 30 m² por defecto.")
    area_visible = 30.0

print(f"\nÁrea visible configurada: {area_visible} m²")
print("→ Se calculará densidad real en personas/m²\n")

# Opcional: área total para extrapolación
try:
    area_total = float(input("¿Área total de la zona que quieres estimar (m²)? (Ej: plaza completa 1000). "
                             "Presiona Enter para omitir: ") or "0")
except:
    area_total = 0

# ============================================
# 2. FUNCIÓN DE CLASIFICACIÓN DE DENSIDAD
# ============================================

def clasificar_densidad(densidad):
    if densidad < 1.0:
        return "baja densidad"
    elif densidad < 2.0:
        return "densidad media"
    elif densidad < 3.0:
        return "alta densidad"
    else:
        return "muy alta densidad (¡atención!)"

# ============================================
# 3. CÁMARA Y CSV
# ============================================

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara")
    exit()

# Ruta CSV en Escritorio
desktop = os.path.join(os.path.expanduser("~"), "Desktop")
csv_path = os.path.join(desktop, f"conteo_personas_densidad_{datetime.now().strftime('%Y%m%d_%H%M')}.csv")

data = []
densidades = []
inicio = time.time()
PROMEDIO_CADA = 10

print("▶ INICIADO - Apunta la cámara y presiona 'q' para detener\n")

# ============================================
# 4. LOOP PRINCIPAL
# ============================================

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.4, classes=[PERSON_CLASS_ID])[0]
        personas = len(results.boxes) if results.boxes is not None else 0

        # Calcular densidad
        densidad = personas / area_visible if area_visible > 0 else 0
        clasificacion = clasificar_densidad(densidad)

        # Dibujar cajas
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        ts = datetime.now().strftime("%H:%M:%S")

        # Mostrar en pantalla
        cv2.putText(frame, f"Personas: {personas}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        cv2.putText(frame, f"Densidad: {densidad:.2f} pers/m²", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
        cv2.putText(frame, clasificacion.upper(), (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
        cv2.putText(frame, f"Hora: {ts}", (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Contador con Densidad - Carnaval Nariño", frame)

        # Guardar datos
        data.append({
            "timestamp": ts,
            "personas": personas,
            "densidad_pers_m2": round(densidad, 3),
            "clasificacion": clasificacion
        })
        densidades.append(densidad)

        # Promedio cada 10 segundos
        if time.time() - inicio >= PROMEDIO_CADA:
            prom_densidad = np.mean(densidades)
            prom_personas = np.mean([d["personas"] for d in data[-300:]])  # aprox últimos 10s
            print(f"Promedio últimos {PROMEDIO_CADA}s: {prom_personas:.1f} personas → "
                  f"{prom_densidad:.2f} pers/m² ({clasificar_densidad(prom_densidad)})")
            densidades = []
            inicio = time.time()

        if cv2.waitKey(1) == ord('q'):
            break

except Exception as e:
    print(f"Error: {e}")

finally:
    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

# ============================================
# 5. GUARDAR CSV Y RESUMEN
# ============================================

if len(data) > 0:
    df = pd.DataFrame(data)

    # Clustering de clasificaciones
    embeddings = text_model.encode(df["clasificacion"].tolist())
    n_clusters = min(4, max(1, len(set(df["clasificacion"]))))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["cluster"] = kmeans.fit_predict(embeddings)

    # Guardar CSV
    df.to_csv(csv_path, index=False)

    # Estadísticas
    densidad_promedio = df["densidad_pers_m2"].mean()
    densidad_max = df["densidad_pers_m2"].max()
    personas_promedio = df["personas"].mean()

    print(f"\n¡Captura finalizada!")
    print(f"CSV guardado en tu Escritorio:")
    print(f"   {csv_path}")
    print(f"   {len(df)} registros capturados")

    print(f"\nRESUMEN DE DENSIDAD:")
    print(f"• Densidad promedio: {densidad_promedio:.2f} personas/m²")
    print(f"• Densidad máxima: {densidad_max:.2f} personas/m²")
    print(f"• Personas promedio visibles: {personas_promedio:.1f}")

    if area_total > 0:
        estimado_total = densidad_promedio * area_total
        print(f"\nESTIMACIÓN EXTRAPOLADA (para {area_total} m²):")
        print(f"→ Aproximadamente {estimado_total:.0f} personas en toda la zona")

    # Gráficos
    plt.figure(figsize=(12,8))
    plt.subplot(2,1,1)
    plt.plot(df["personas"], color="blue", linewidth=2)
    plt.title("Conteo de personas por frame")
    plt.ylabel("Personas")
    plt.grid(True, alpha=0.3)

    plt.subplot(2,1,2)
    plt.plot(df["densidad_pers_m2"], color="red", linewidth=2)
    plt.title("Densidad real (personas/m²)")
    plt.xlabel("Tiempo (frames)")
    plt.ylabel("Densidad (pers/m²)")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

else:
    print("No se capturaron datos.")
