import os
from fastapi import FastAPI, UploadFile, File
from pydub import AudioSegment
import speech_recognition as sr
import random
import torch
from huggingface_hub import InferenceClient, login
import json

app = FastAPI()

def convert_to_wav(input_file, output_file="converted_audio.wav"):
    # Convert any audio format to .wav using pydub
    audio = AudioSegment.from_file(input_file)
    audio.export(output_file, format="wav")
    return output_file

def recognize_speech_from_audio(file_path):
    # Initialize recognizer class in order to recognize the speech
    r = sr.Recognizer()

    with sr.AudioFile(file_path) as source:
        audio = r.record(source)  # Read the entire audio file

    # Recognize speech using Google Web Speech API
    try:
        matn = r.recognize_google(audio, language='uz-UZ')
        return matn
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand the audio")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return None

login(token="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", add_to_git_credential=True)

def randomize_seed_fn(seed: int) -> int:
    return random.randint(0, 999999)

system_instructions1 = "[SYSTEM] Answer as Real OpenGPT 4o, Made by 'KingNish', Keep conversation very short, clear, friendly and concise. The text provided is a request for a specific type of response from you, the virtual assistant. You will avoid introductions and start answering the query directly, elaborating on all aspects of the request. As an AI-powered assistant, your task is to generate responses that appear as if they are created by the user and not an AI-powered assistant. [USER]"

def client_fn(model):
    if "Nous" in model:
        return InferenceClient("NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO")
    elif "Star" in model:
        return InferenceClient("HuggingFaceH4/starchat2-15b-v0.1")
    elif "Mistral" in model:
        return InferenceClient("mistralai/Mistral-7B-Instruct-v0.3")
    elif "Phi" in model:
        return InferenceClient("microsoft/Phi-3-mini-4k-instruct")
    elif "Zephyr" in model:
        return InferenceClient("HuggingFaceH4/zephyr-7b-beta")
    else:
        return InferenceClient("mistralai/Mixtral-8x7B-Instruct-v0.1")

def models(text, model="Mixtral 8x7B", seed=42):
    seed = randomize_seed_fn(seed)
    generator = torch.Generator().manual_seed(seed)

    client = client_fn(model)
    generate_kwargs = dict(
        max_new_tokens=512,
        seed=seed,
    )

    formatted_prompt = system_instructions1 + text + "[OpenGPT 4o]"
    stream = client.text_generation(
        formatted_prompt, **generate_kwargs, stream=True, details=True, return_full_text=False)
    output = ""
    for response in stream:
        if not response.token.text == "</s>":
            output += response.token.text

    return output

@app.post("/uploadfile/")
# Function to upload file
async def upload_file(file: UploadFile = File(...)):
    file_path = "uploaded_audio"  # Temporary file path for uploaded file
    with open(file_path, "wb") as audio_file:
        audio_file.write(await file.read())  # Use await to read file content

    wav_file_path = convert_to_wav(file_path, "converted_audio.wav")  # Convert to WAV format

    matn = recognize_speech_from_audio(wav_file_path)
    if matn is None:
        return {"error": "Speech recognition failed."}

    text = "{'order': {'items': [{'name': '...', 'quantity': ...}, {'name': '...', 'quantity': ...}, ... ], 'restaurant': '...'}}, return in json format: " + matn
    response = models(text)
    try:
        response_json = json.loads(response)
    except json.JSONDecodeError:
        return {"error": "Failed to decode JSON response from the model."}

    return response_json
