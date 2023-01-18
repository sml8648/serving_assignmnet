from fastapi import FastAPI
from Dataloader import ReturnData
from Model import Model

app = FastAPI()
returndata = ReturnData()
model = Model()

@app.get("/")
async def root():
    return {"message": "Hello AI serving assignment"}

@app.get("/inference")
async def inference(sentence1 : str, sentence2: str):

    tokenized_data = returndata.tokenizing(sentence1, sentence2)
    output = model(tokenized_data)
    
    return {
        "Return_score" : float(output.detach().cpu().numpy())
    }