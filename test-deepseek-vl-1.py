import torch
from PIL import Image
from transformers import AutoModelForCausalLM

from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images
# HuggingFace의 transformers가 아님

# 모델 경로
model_path = "deepseek-ai/deepseek-vl-1.3b-chat"

# 모델 & 프로세서 로드
print("모델 로드")
model = MultiModalityCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to("cuda").eval()
print("프로세서 로드")
processor = VLChatProcessor.from_pretrained(model_path)
tokenizer = processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

# image = Image.open("F:/image/receipt/r (1).jpg").convert('RGB')
prompt = "Sort by category (refrigerated, frozen), item, quantity, price, and delivery date."

image_path = "F:/image/receipt/online/2.png"

conversation = [
    {
        "role": "|User|",
        "content": f"<image_placeholder>{prompt}.",
        "images": [image_path],
    },
    {"role": "Assistant", "content": ""},
]

pil_images = load_pil_images(conversation)

# load images and prepare for inputs
print("입력생성")

prepare_inputs = processor(
    conversations=conversation,
    images=pil_images,
    # images=[image],
    force_batchify=True,
    system_prompt=""
).to(vl_gpt.device)

print("임베딩")
inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

# run the model to get the response
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

answer = tokenizer.decode(outputs[0].tolist(), skip_special_tokens=True)
print(f"{prepare_inputs['sft_format'][0]}", answer)