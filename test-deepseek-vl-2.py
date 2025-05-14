import torch
from PIL import Image
from transformers import AutoModelForCausalLM
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())

# specify the path to the model
print("LoadModel")
model_path = "deepseek-ai/deepseek-vl-1.3b-chat"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

print("LoadModal")
vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

## single image conversation example
conversation = [
    {
        "role": "User",
        #"content": "<image_placeholder>Sort by category (refrigerated, frozen), item, quantity, price, and delivery date.",
        "content": "<image_placeholder>Describe the image.",
        "images": ["F:/image/receipt/online/2.png"],
    },
    {"role": "Assistant", "content": ""},
]

# load images and prepare for inputs
pil_images = load_pil_images(conversation)

print("Loaded image count:", len(pil_images))
for idx, img in enumerate(pil_images):
    print(f"Image {idx} size:", img.size, "mode:", img.mode)

prepare_inputs = vl_chat_processor(
    conversations=conversation,
    images=pil_images,
    force_batchify=True
).to(vl_gpt.device)

# run image encoder to get the image embeddings
inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

print("inputs_embeds shape:", inputs_embeds.shape)

# run the model to get the response
with torch.no_grad():
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False, #False
        use_cache=True
    )

full_text = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
# 'Assistant:' 다음 부분만 추출
if "Assistant:" in full_text:
    answer = full_text.split("Assistant:")[-1].strip()
else:
    answer = full_text.strip()
print("Answer:", answer)