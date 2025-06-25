from fastapi import FastAPI
from pydantic import BaseModel
from intent_classifier import IntentClassifier
from response_generator import GPT2Responder

# Initialize app and models
app = FastAPI()
classifier = IntentClassifier()
responder = GPT2Responder()

# Data model for request
class Query(BaseModel):
    message: str

@app.post("/chat/")
async def chat(query: Query):
    user_message = query.message
    intent = classifier.predict_intent(user_message)
    prompt = f"Intent: {intent}\nUser: {user_message}\nBot:"
    response = responder.generate_response(prompt)
    return {"intent": intent, "response": response}
