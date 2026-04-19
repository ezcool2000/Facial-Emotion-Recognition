import cv2
from deepface import DeepFace

# Kamerayı başlat
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Hata: Kamera açılamadı.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Hata: Görüntü alınamıyor.")
        break

    try:
        results = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False
        )

        for face in results:
            x = face['region']['x']
            y = face['region']['y']
            w = face['region']['w']
            h = face['region']['h']
            emotion = face['dominant_emotion']

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(
                frame,
                emotion,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (36, 255, 12),
                2
            )

    except Exception as e:
        print("Hata:", e)

    cv2.imshow("DeepFace Duygu Tespiti", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
