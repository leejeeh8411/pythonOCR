from transformers import DonutProcessor, VisionEncoderDecoderModel
from datasets import load_dataset
import torch
import random
from PIL import Image
import cv2
import numpy as np

def preprocess_image(pil_image):
    # PIL → OpenCV BGR
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # BGR → Grayscale
    gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    # Grayscale → RGB (Donut은 3채널 이미지만 받음)
    rgb_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)

    # 다시 PIL로
    pil_rgb = Image.fromarray(rgb_image)

    return pil_rgb

# 1. 설정
model_checkpoint = "naver-clova-ix/donut-base"
task_prompt = "<s_synthdog-ko>" #"<s_synthdog-ko>"  # 학습 시 사용한 task prompt
max_length = 768
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 모델 및 프로세서 로드
processor = DonutProcessor.from_pretrained(model_checkpoint)
model = VisionEncoderDecoderModel.from_pretrained(model_checkpoint)
model.to(device)
model.eval()

# 3. 데이터셋 로드
dataset = load_dataset("naver-clova-ix/synthdog-ko", split="train")

# 4. 랜덤으로 이미지 하나 선택
sample = random.choice(dataset)
image = sample["image"]
image = Image.open("F:/image/receipt/r (3).jpg")
# image = Image.open("F:/image/receipt/online/2.png")
# image = preprocess_image(image)

# 5. 이미지 전처리 (pixel_values 생성)
pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

# 6. 추론 (텍스트 생성)
# prompt를 사용할 경우, decoder_input_ids를 명시적으로 설정할 수 있습니다.
decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

with torch.no_grad():
    generated_ids = model.generate(
        pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=max_length,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
    )

# 7. 결과 디코딩
generated_text = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

# 8. 프롬프트 제거 및 출력
if generated_text.startswith(task_prompt):
    generated_text = generated_text[len(task_prompt):]

print("추론 결과:", generated_text.strip())

image.show()