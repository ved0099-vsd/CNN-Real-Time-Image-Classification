# ============================================================
# Project  : CNN - Real-Time Image Classification
# Author   : Vedant Dhamal
# ============================================================

import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

def VedImageClassifier():
    # 1) Load pre-trained CNN (ImageNet)
    model = MobileNetV2(weights="imagenet")

    # 2) Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # 3) Preprocess for MobileNetV2
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (224, 224))
        x = np.expand_dims(img_resized, axis=0).astype(np.float32)
        x = preprocess_input(x)

        # 4) Predict
        preds = model.predict(x, verbose=0)
        decoded = decode_predictions(preds, top=1)[0][0]
        label = f"{decoded[1]}: {decoded[2]*100:.1f}%"

        # 5) Overlay prediction
        cv2.putText(frame, label, (16, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Ved Real-time CNN Classification", frame)

        # 6) Exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 7) Cleanup
    cap.release()
    cv2.destroyAllWindows()


def main():
    VedImageClassifier()


if __name__ == "__main__":
    main()