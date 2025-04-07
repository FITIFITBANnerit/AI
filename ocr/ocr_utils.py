import cv2
import numpy as np

import cv2
import numpy as np

class OCRPreprocessing:
    def __init__(self, image, clipLimit=2.0, tileGridSize=(8,8), kernel=None):
        self.image = image
        self.clipLimit = clipLimit
        self.tileGridSize = tileGridSize
        self.kernel = kernel if kernel is not None else np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    def converse_hsv(self, image):
        """ HSV 변환 후 배경 제거 """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)
        return cv2.bitwise_and(image, image, mask=~mask)

    def gray_scale(self, image):
        """ Grayscale 변환 """
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def histogram_clahe(self, image):
        """ CLAHE 적용 (대비 증가) """
        clahe = cv2.createCLAHE(self.clipLimit, self.tileGridSize)
        return clahe.apply(image)

    def sharpening(self, image):
        """ 샤프닝 적용 """
        return cv2.filter2D(image, -1, self.kernel)

    def remove_shadow(self, image):
        """ 그림자 제거 """
        dilated_img = cv2.dilate(image, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(image, bg_img)
        return cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    def invert_colors(self, image):
        """ 색 반전 (배경이 어두울 경우) """
        return cv2.bitwise_not(image)

    def apply_threshold(self, image):
        """ Otsu's Thresholding 적용 """
        _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary_image

    def image_preprocessing(self):
        """ 전체 전처리 과정 실행 """
        img = self.converse_hsv(self.image)     # 배경 제거
        img = self.gray_scale(img)              # Grayscale 변환
        img = self.histogram_clahe(img)         # CLAHE 적용 (대비 증가)
        img = self.sharpening(img)              # 샤프닝 적용 (필요할 경우)
        img = self.remove_shadow(img)           # 그림자 제거
        img = self.invert_colors(img)           # 색 반전 (배경이 어두운 경우)
        img = self.apply_threshold(img)         # Thresholding 적용 (OCR 최적화)
        return self.image
