
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import torch
from fastapi import FastAPI, Request
import uvicorn
import asyncio
import base64
from io import BytesIO
import time

bf16 = False
processor = None
model = None
lock = asyncio.Lock()

def decode_base64_to_pil_image(encoded_string):
    decoded_bytes = base64.b64decode(encoded_string)
    image = Image.open(BytesIO(decoded_bytes))
    return image

def load_model():
    global processor, model, bf16

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
def generate(image_base64, text):
    global processor, model, bf16
    
    image = decode_base64_to_pil_image(image_base64)
    # process the image and text
    inputs = processor.process(
        images=[ image ],
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
    async with lock:
        # only one request at a time
        before = time.time()
        res = generate(data['image_base64'], data['text'])
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
    test_model()
    
    