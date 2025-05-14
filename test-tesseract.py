from PIL import Image, ImageEnhance
import pytesseract  # OCR 엔진
import cv2
import numpy as np
from matplotlib import pyplot as plt

def preprocess_image(image_path):
    # 1. 이미지 읽기 (흑백으로 로드)
    img = cv2.imread(image_path)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)  # 노이즈 감소
    dilated = cv2.dilate(gray, np.ones((7,7), np.uint8))
    bg = cv2.medianBlur(dilated, 21)
    result = 255 - cv2.absdiff(gray, bg)
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)

    return result

# 이미지 경로 지정
image_path = "F:/image/receipt/online/1.png"
image_path = "F:/image/receipt/r (1).jpg"

img = cv2.imread(image_path)
# # 전처리 적용
# result = preprocess_image(image_path)

# thresh = cv2.adaptiveThreshold(
#     result, 255,
#     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#     cv2.THRESH_BINARY,
#     31, 10
# )

# rotated = thresh
# # coords = np.column_stack(np.where(thresh > 0))
# # angle = cv2.minAreaRect(coords)[-1]
# # if angle < -45:
# #     angle = -(90 + angle)
# # else:
# #     angle = -angle

# # (h, w) = thresh.shape[:2]
# # center = (w // 2, h // 2)
# # M = cv2.getRotationMatrix2D(center, angle, 1.0)
# # rotated = cv2.warpAffine(thresh, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

# contours, _ = cv2.findContours(rotated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# for c in contours:
#     x, y, w, h = cv2.boundingRect(c)
#     if w < 50 or h < 20:  # 너무 작은 요소 제거
#         cv2.rectangle(rotated, (x, y), (x + w, y + h), (0, 0, 0), -1)

# cv2.imwrite("F:/image/receipt/pre.jpg", rotated)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Windows인 경우

# 전처리된 이미지로 OCR 실행
text = pytesseract.image_to_string(
    img, 
    lang='kor', 
    config = r'--oem 3 --psm 3 -l kor'  # OEM 3(자동), PSM 11(밀집 텍스트)
)

# text = pytesseract.image_to_string(
#     img, 
#     lang='kor+eng', 
#     config = r'--oem 3 --psm 4 -l kor+eng'  # OEM 3(자동), PSM 11(밀집 텍스트)
# )
print("추출된 텍스트:\n", text)
