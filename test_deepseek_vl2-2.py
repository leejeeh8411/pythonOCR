import torch
from transformers import AutoModelForCausalLM

from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images

# 모델 경로
model_path = "deepseek-ai/deepseek-vl2-tiny"

# Processor 및 tokenizer 로드
vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

# 모델 로드 (CPU, float32)
vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.float32).to("cpu").eval()

# 대화 예시
conversation = [
    {
        "role": "<|User|>",
        "content": "<image>\n<|ref|>what is this?<|/ref|>.",
        "images": ["F:/image/receipt/online/2.png"],
    },
    {"role": "<|Assistant|>", "content": ""},
]

# 이미지 로딩 및 입력 준비
pil_images = load_pil_images(conversation)
prepare_inputs = vl_chat_processor(
    conversations=conversation,
    images=pil_images,
    force_batchify=True,
    system_prompt=""
).to("cpu")  # CPU로 입력 이동

# 이미지 임베딩 준비
inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

# 텍스트 생성
outputs = vl_gpt.language.generate(
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
answer = tokenizer.decode(outputs[0].tolist(), skip_special_tokens=False)
print(f"{prepare_inputs['sft_format'][0]}", answer)
