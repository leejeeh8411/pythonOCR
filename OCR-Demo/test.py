from transformers import DonutProcessor, VisionEncoderDecoderModel
from datasets import load_dataset
import torch
from PIL import Image
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 및 프로세서 로드
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2").to(device)

# 데이터셋 로드
# dataset = load_dataset("naver-clova-ix/cord-v2", split="train")
dataset = load_dataset("naver-clova-ix/synthdog-ko", split="train")

# 랜덤 인덱스 선택
random_idx = random.randint(0, len(dataset) - 1)

image = dataset[random_idx]["image"]
ground_truth = dataset[random_idx]["ground_truth"]
# image = Image.open("F:/image/receipt/online/2.png")
# image = Image.open("F:/image/receipt/r (3).jpg")

# 모델 입력 전처리
pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

# 디코딩용 prompt (Donut은 JSON 형태로 추출)
task_prompt = "<s_cord-v2>"  # 모델이 사용하는 프롬프트 토큰
decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
decoder_input_ids = decoder_input_ids.to(device)

# 추론
outputs = model.generate(pixel_values, decoder_input_ids=decoder_input_ids, max_length=768)
result = processor.batch_decode(outputs, skip_special_tokens=True)[0]

print("📄 정답:")
print(ground_truth)

print("📄 Donut OCR 인식 결과:")
print(result)
image.show()
