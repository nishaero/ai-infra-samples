import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import uvicorn
import transformers
from app.registry import get_current_model, switch_model

app = FastAPI()
MODEL, MODEL_INFO = get_current_model()

# Hugging Face token from environment
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

model = None
tokenizer = None

def load_model(model_id):
    global model, tokenizer
    print(f"Loading model: {model_id}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
        use_auth_token=HF_TOKEN if HF_TOKEN else None
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        use_auth_token=HF_TOKEN if HF_TOKEN else None
    )

load_model(MODEL)

@app.post("/generate")
async def generate(data: dict):
    prompt = data.get("prompt")
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing prompt")
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=128)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return JSONResponse({"response": response})

@app.post("/set-model")
async def set_model(data: dict, request: Request):
    # TODO: Add JWT auth here
    new_model_id = data.get("model_id")
    if not new_model_id:
        raise HTTPException(status_code=400, detail="Missing model_id")
    switch_model(new_model_id)
    load_model(new_model_id)
    return JSONResponse({"status": "ok", "new_model": new_model_id})

@app.get("/model-info")
async def model_info():
    return JSONResponse(MODEL_INFO)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
