import sys
import torch
from PIL import Image
import numpy as np
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images

# PyTorch 버전 및 CUDA 사용 가능 여부 출력
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("Device count:", torch.cuda.device_count())
    print("Device name:", torch.cuda.get_device_name(0))
else:
    print("This PyTorch build does NOT support CUDA (CPU-only).")

# CUDA 장치 사용 여부 확인
if torch.cuda.is_available():
    print("쿠다 사용")
    device = torch.device("cuda")
else:
    print("CPU 사용")
    device = torch.device("cpu")

# sys.exit()

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,     # 또는 load_in_4bit=True
    llm_int8_threshold=6.0,
    llm_int8_skip_modules=None,  # 특정 모듈 제외 가능
    llm_int8_enable_fp32_cpu_offload=True,
)


# 모델 경로 설정
model_path = "deepseek-ai/deepseek-vl2-tiny"
vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

# 모델 & 프로세서 로드
print("모델 로드")
vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
# vl_gpt: DeepseekVLV2ForCausalLM = DeepseekVLV2ForCausalLM.from_pretrained(
    model_path, 
    trust_remote_code=True,
    # quantization_config=bnb_config, # 양자화 옵션
    # device_map="auto"  # GPU에 자동 배치
    # torch_dtype=torch.float16
    )
#vl_gpt = vl_gpt.cuda().eval()
#vl_gpt = vl_gpt.half().cuda().eval()
print("[eval]")
vl_gpt.eval()

# 프롬프트 설정
prompt = "Sort by category (refrigerated, frozen), item, quantity, price, and delivery date."

# 대화 내용 설정
conversation = [
    {
        "role": "<|User|>",
        "content": f"<image>\n<|ref|>{prompt}<|/ref|>.",
        "images": ["F:/image/receipt/online/2.png"],
    },
    {"role": "<|Assistant|>", "content": ""},
]

print("[Load Image]")
# 이미지를 PIL 형식으로 로드
pil_images = load_pil_images(conversation)

# 입력 데이터 준비
print("입력 생성")
prepare_inputs = vl_chat_processor(
    conversations=conversation,
    images=pil_images,
    force_batchify=True,
    system_prompt=""
)

# prepare_inputs를 dict로 변환
prepare_inputs = dict(prepare_inputs)

# device로 옮길 수 있는 항목만 개별적으로 처리
for k, v in prepare_inputs.items():
    if isinstance(v, torch.Tensor):
        prepare_inputs[k] = v.to(device)

# 텐서 타입 변환 및 device 이동
# for k, v in prepare_inputs.items():
#     if isinstance(v, torch.Tensor):
#         # float 타입이면 float16으로 변환 (2080Ti는 bfloat16 미지원)
#         if v.dtype == torch.bfloat16:
#             prepare_inputs[k] = v.to(torch.float16)
#         else:
#             prepare_inputs[k] = v.to(device)

# batch_num_tiles가 Tensor면 list[int]로 변환 (prepare_inputs_embeds 요구사항)
# if "batch_num_tiles" in prepare_inputs and isinstance(prepare_inputs["batch_num_tiles"], torch.Tensor):
#     prepare_inputs["batch_num_tiles"] = prepare_inputs["batch_num_tiles"].tolist()



# 모델에 필요한 임베딩 생성
print("임베딩")
inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

# 명시적으로 float16으로 (일관성 유지)
# inputs_embeds = inputs_embeds.to(torch.float16)

# 모델 추론
print("추론")
outputs = vl_gpt.language_model.generate(
    inputs_embeds=inputs_embeds,
    attention_mask=prepare_inputs.attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=512,
    do_sample=False,
    use_cache=True
)

# 결과 디코딩
answer = tokenizer.decode(outputs[0].tolist(), skip_special_tokens=True)
print(f"{prepare_inputs['sft_format'][0]}", answer)
