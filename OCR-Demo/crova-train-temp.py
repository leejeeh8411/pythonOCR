from transformers import DonutProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset
from torch.utils.data import Dataset
import torch
import os

# 하이퍼파라미터
model_checkpoint = "naver-clova-ix/donut-base-finetuned-cord-v2"
task_prompt = "<s_synthdog-ko>"  # 커스텀 프롬프트 설정
max_target_length = 768
max_source_length = 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 모델과 프로세서 불러오기
processor = DonutProcessor.from_pretrained(model_checkpoint)
model = VisionEncoderDecoderModel.from_pretrained(model_checkpoint)
model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(task_prompt)
model.config.pad_token_id = processor.tokenizer.pad_token_id  # 추가: 패딩 토큰 ID 설정
model.to(device)

# 2. 커스텀 PyTorch Dataset 클래스 정의
class SynthDogDataset(Dataset):
    def __init__(self, hf_dataset, processor, task_prompt):
        self.dataset = hf_dataset
        self.processor = processor
        self.prompt = task_prompt

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample["image"]
        text = self.prompt + sample["ground_truth"]

        # 입력 이미지 전처리
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()

        # 출력 텍스트 토크나이즈
        labels = self.processor.tokenizer(
            text,
            max_length=max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze()

        # padding token -> -100 (loss 계산 시 무시)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels
        }

# 3. 데이터셋 로드
# raw_dataset = load_dataset("naver-clova-ix/synthdog-ko", split="train")
# 처음 1000개만 학습 (빠른 테스트용)
dataset = load_dataset("naver-clova-ix/synthdog-ko", split="train[:20000]")
dataset = dataset.train_test_split(test_size=0.1)
train_dataset = SynthDogDataset(dataset["train"], processor, task_prompt)
eval_dataset = SynthDogDataset(dataset["test"], processor, task_prompt)
# dataset = SynthDogDataset(dataset, processor, task_prompt)

# 4. TrainingArguments 정의
training_args = Seq2SeqTrainingArguments(
    output_dir="./donut-synthdog-ko",
    per_device_train_batch_size=1,
    num_train_epochs=1, # 3
    gradient_accumulation_steps=2,
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    learning_rate=5e-5,
    # evaluation_strategy="steps",  #추가
    evaluation_strategy="no",  #추가
    eval_steps=200, #추가
)

def collate_fn(batch):
    pixel_values = torch.stack([example["pixel_values"] for example in batch])
    labels = torch.stack([example["labels"] for example in batch])
    return {"pixel_values": pixel_values, "labels": labels}

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    decoded_preds = processor.batch_decode(pred_ids, skip_special_tokens=True)
    decoded_labels = processor.batch_decode(label_ids, skip_special_tokens=True)

    # Character-level accuracy
    correct = 0
    total = 0
    for pred_str, label_str in zip(decoded_preds, decoded_labels):
        min_len = min(len(pred_str), len(label_str))
        correct += sum([p == l for p, l in zip(pred_str, label_str)])
        total += max(len(pred_str), len(label_str))
    accuracy = correct / total if total > 0 else 0.0

    return {"char_accuracy": accuracy}

# 5. Trainer 정의
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor.tokenizer,
    data_collator=collate_fn,  # 데이터 콜레이터 사용
    compute_metrics=compute_metrics
)

# 6. 학습 시작
trainer.train()

# 7. 저장
model.save_pretrained("./donut-synthdog-ko")
processor.save_pretrained("./donut-synthdog-ko")
