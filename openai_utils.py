import os
import openai
import keyboard
import pyautogui
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav

# Set the duration of the recording here
DURATION = 10  # In seconds

# Set the hotkey here
HOTKEY = "ctrl+alt+a"

# Set the path to save the audio file here
AUDIO_PATH = "temp/audio.wav"

# Get OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")


# Function to record audio
def record_audio(duration, path):
    fs = 44100
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=2)
    sd.wait()
    wav.write(path, fs, np.int16(recording))


# Function to transcribe audio
def transcribe_audio(path):
    with open(path, "rb") as audio_file:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript["text"]


# Function to type text
def type_text(text):
    pyautogui.write(text)


# Function to handle hotkey press
def on_hotkey():
    print("Hotkey pressed, recording audio...")
    record_audio(DURATION, AUDIO_PATH)
    print("Audio recorded, transcribing...")
    text = transcribe_audio(AUDIO_PATH)
    print("Audio transcribed, typing text...")
    type_text(text)
    print("Text typed.")


# Listen for hotkey
keyboard.add_hotkey(HOTKEY, on_hotkey)
print("Listening for hotkey...")
keyboard.wait()
