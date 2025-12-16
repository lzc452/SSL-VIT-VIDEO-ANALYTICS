import cv2
import numpy as np


class YuNetFaceDetector:
    def __init__(self, model_path, conf_th=0.6, nms_th=0.3):
        self.detector = cv2.FaceDetectorYN.create(
            model_path,
            "",
            (320, 320),
            conf_th,
            nms_th,
            5000
        )

    def detect(self, img):
        h, w = img.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(img)
        if faces is None:
            return []
        return faces[:, :4].astype(int)


class VisualAnonymizer:
    def __init__(self, detector: YuNetFaceDetector, method="face_blur", blur_kernel=31):
        self.detector = detector
        self.method = method
        self.blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1

    def apply(self, img):
        faces = self.detector.detect(img)
        out = img.copy()

        for (x, y, w, h) in faces:
            roi = out[y:y+h, x:x+w]
            if roi.size == 0:
                continue
            roi = cv2.GaussianBlur(roi, (self.blur_kernel, self.blur_kernel), 0)
            out[y:y+h, x:x+w] = roi

        return out, len(faces)
