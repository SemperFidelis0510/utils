import os
import datetime
import json
from gtts import gTTS
from googletrans import Translator
import speech_recognition as sr
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, TrainingArguments, Trainer
from datasets import load_dataset
from pydub.playback import play
from pydub import AudioSegment
import tempfile
import random
import simpleaudio as sa


class SpeechRecognition:
    def __init__(self, dataset_dir='./datasets/my_voice'):
        self.recognizer = sr.Recognizer()
        self.dataset_dir = dataset_dir
        os.makedirs(dataset_dir, exist_ok=True)

    def activate_voice_command(self):
        text = ''
        with sr.Microphone() as source:
            print("Listening...")
            audio = self.recognizer.listen(source)
            try:
                text = self.recognizer.recognize_google(audio)
                print("You said: {}".format(text))
            except sr.UnknownValueError:
                print("Google Speech Recognition could not understand audio")
            except sr.RequestError as e:
                print("Could not request results from Google Speech Recognition service; {0}".format(e))
        return text

    def create_dataset(self, sentences_file='sentences.json', record=True):
        n = 10
        sentences_file = os.path.join(self.dataset_dir, sentences_file)
        if os.path.exists(sentences_file):
            with open(sentences_file, 'r', encoding='utf-8') as f:
                sentences = json.load(f)
        else:
            sentences = self.get_random_sentences(10)
            sentence_dicts = [{"sentence": sentence} for sentence in sentences]
            with open(sentences_file, 'w') as f:
                json.dump(sentence_dicts, f)

        if record:
            dataset = []
            temp_dir = os.path.join(self.dataset_dir, 'temp')
            os.makedirs(temp_dir, exist_ok=True)  # Create the temporary directory if it doesn't exist

            for sentence_dict in sentences:
                sentence = sentence_dict['sentence']
                print(f"Say: {sentence}")
                tts = gTTS(sentence, lang="en")
                with tempfile.NamedTemporaryFile(delete=False, dir=temp_dir, suffix='.mp3') as fp:
                    temp_filename = fp.name
                    tts.save(temp_filename)
                    segment = AudioSegment.from_file(temp_filename, format="mp3")
                    play(segment)
                    recorded_audio = self.record_sentence(sentence)
                    filename = self.save_wav_file(sentence, recorded_audio)
                    dataset.append({"path": filename, "transcription": sentence})

            # Save the dataset to a JSON file
            dataset_file = os.path.join(self.dataset_dir, 'dataset.json')
            with open(dataset_file, 'w') as f:
                json.dump(dataset, f)

    def get_random_sentences(self, n):
        sentences_file = os.path.join(self.dataset_dir, 'sentences.txt')
        with open(sentences_file, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f]

        random_sentences = random.sample(sentences, n)
        return random_sentences

    def save_wav_file(self, sentence, recorded_audio):
        filename = f"{hash(sentence)}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        filename = os.path.join(self.dataset_dir, filename)
        with open(filename, "wb") as f:
            f.write(recorded_audio.get_wav_data())

    def record_sentence(self, sentence):
        mic_index = 1
        with sr.Microphone(device_index=mic_index) as source:
            print(f"Say: {sentence}")
            audio = self.recognizer.listen(source)
        return audio


def main():
    get_data = True
    record = True
    # Initialize the SpeechRecognition class
    speech_recognition = SpeechRecognition()

    # Create the dataset
    if get_data:
        speech_recognition.create_dataset(record=record)

    # Define the path to the dataset
    dataset_path = './datasets/my_voice/sentences.json'

    # Fine-tune the model
    # SpeechRecognition.fine_tune_model(dataset_path)


if __name__ == "__main__":
    main()
