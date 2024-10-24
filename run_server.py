
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import torch
from fastapi import FastAPI, Request
import uvicorn
import asyncio
import base64
from io import BytesIO
import time
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

bf16 = False
processor = None
model = None
lock = asyncio.Lock()
model_name = "qwen_7B"

def decode_base64_to_pil_image(encoded_string):
    decoded_bytes = base64.b64decode(encoded_string)
    image = Image.open(BytesIO(decoded_bytes))
    return image

def load_model():

    global processor, model, bf16, model_name

    if model_name == "molmo":
        bf16 = True
        # load the processor
        processor = AutoProcessor.from_pretrained(
            'allenai/Molmo-7B-D-0924',
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto'
        )

        # load the model
        model = AutoModelForCausalLM.from_pretrained(
            'allenai/Molmo-7B-D-0924',
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto'
        )

        if bf16:
            model.to(dtype=torch.bfloat16)
    
    elif model_name == "qwen_72B":
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int4", torch_dtype="auto", device_map="auto"
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int4")
    elif model_name == "qwen_7B":
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4", torch_dtype="auto", device_map="auto"
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4")

    else:
        raise ValueError(f"Unknown model name: {model_name}")

@torch.no_grad()
def generate_batch(image_base64_list, text_list):
    global processor, model, bf16
    
    images = [decode_base64_to_pil_image(image_base64) for image_base64 in image_base64_list]
    # process the image and text
    inputs = processor.process(
        images=images,
        text=text_list,
    )

    # move inputs to the correct device 
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    if bf16:
        inputs["images"] = inputs["images"].to(torch.bfloat16)

    with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
        # generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
        output = model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=2000, stop_strings="<|endoftext|>"),
            tokenizer=processor.tokenizer
        )
    
    # only get generated tokens; decode them to text
    generated_text_list = []
    for i in range(len(image_base64_list)):
        generated_tokens = output[i,inputs['input_ids'].size(1):]
        generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        generated_text_list.append(generated_text)

    # print the generated text
    print(generated_text)

    return generated_text_list

@torch.no_grad()
def generate(image_base64, text, new_model_name=None):
    global processor, model, bf16, model_name
    if new_model_name is not None:
        model_name = new_model_name
        load_model()
    
    if model_name == "qwen_72B" or model_name == "qwen_7B":
    
        content = [{"type":"image", "image": decode_base64_to_pil_image(image)} for image in image_base64]
        content.append({"type":"text", "text": text})
        # Messages containing multiple images and a text query
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]

        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference
        generated_ids = model.generate(**inputs, max_new_tokens=1280)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(output_text)
        generated_text = output_text[0]


    elif model_name == "molmo":
    
        images = [decode_base64_to_pil_image(image) for image in image_base64]
        # process the image and text
        inputs = processor.process(
            images=images ,
            text=text,
        )

        # move inputs to the correct device and make a batch of size 1
        inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

        if bf16:
            inputs["images"] = inputs["images"].to(torch.bfloat16)

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            # generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
            output = model.generate_from_batch(
                inputs,
                GenerationConfig(max_new_tokens=2000, stop_strings="<|endoftext|>"),
                tokenizer=processor.tokenizer
            )
        
        # only get generated tokens; decode them to text
        generated_tokens = output[0,inputs['input_ids'].size(1):]
        generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # print the generated text
        print(generated_text)

    return generated_text



app = FastAPI()

@app.post("/generate-batch")
async def handle_generate_batch(request: Request):
    data = await request.json()
    async with lock:
        # only one request at a time
        before = time.time()
        res = generate_batch(data['image_base64'], data['text'])
        print(f"Time taken: {time.time() - before}")
    print(res)
    
    return {"response": res}

@app.post("/generate")
async def handle_generate(request: Request):
    data = await request.json()
    print(f'Processing request: {data}')
    async with lock:
        # only one request at a time
        before = time.time()
        res = generate(data['image_base64'], data['text'], data.get("model_name",None))
        print(f"Time taken: {time.time() - before}")
    print(res)
    
    return {"response": res}


def test_model():
    prompt = "The weather is"
    res = generate(prompt)
    print(res)

if __name__ == "__main__":
    load_model()
    uvicorn.run(app, host="localhost", port=8000)
    # test_model()
    
    