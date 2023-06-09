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
            with open(sentences_file, 'r') as f:
                sentences = json.load(f)
        else:
            sentences = self.generate_random_sentences(10)
            sentence_dicts = [{"sentence": sentence} for sentence in sentences]
            with open(sentences_file, 'w') as f:
                json.dump(sentence_dicts, f)

        if record:
            for sentence in sentences:
                # Save gTTS object to a temporary file
                fd, path = tempfile.mkstemp()
                try:
                    with os.fdopen(fd, 'w') as tmp:
                        tts = gTTS(sentence, lang="en")
                        tts.save(path)
                        # Load temporary file into an AudioSegment
                        segment = AudioSegment.from_file(path, format="mp3")
                        # Convert to wav
                        wav_data = segment.export(format="wav")
                        # Play the audio using simpleaudio
                        wave_obj = sa.WaveObject.from_wave_file(wav_data)
                        play_obj = wave_obj.play()
                        play_obj.wait_done()
                        recorded_audio = self.record_sentence(sentence)
                        self.save_wav_file(sentence, recorded_audio)
                finally:
                    os.remove(path)  # Delete the temporary file

    @staticmethod
    def fine_tune_model(dataset_path):
        dataset = load_dataset('json', data_files=dataset_path)
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base")
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

        def prepare_dataset(batch):
            input_values = processor(batch["path"], sampling_rate=16000, return_tensors="pt", padding=True,
                                     max_length=1024).input_values
            with processor.as_target_processor():
                labels = processor(batch["transcription"], return_tensors="pt", padding=True).input_ids
            return {"input_values": input_values, "labels": labels}

        preprocessed_dataset = dataset.map(prepare_dataset, batched=True, remove_columns=["path", "transcription"])

        training_args = TrainingArguments(
            output_dir="./output",
            num_train_epochs=3,
            per_device_train_batch_size=8,
            learning_rate=5e-5,
            weight_decay=0.01,
            logging_dir="./logs"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=preprocessed_dataset["train"],
            data_collator=lambda data: {"input_values": data[0]["input_values"], "labels": data[0]["labels"]}
        )

        trainer.train()
        model.save_pretrained("./output")

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
        # List all available microphones
        # print(sr.Microphone.list_microphone_names())

        mic_index = 1
        with sr.Microphone(device_index=mic_index) as source:
            print(f"Say: {sentence}")
            audio = self.recognizer.listen(source)
        return audio


def main():
    get_data = False
    # Initialize the SpeechRecognition class
    speech_recognition = SpeechRecognition()

    # Create the dataset
    if get_data:
        speech_recognition.create_dataset()

    # Define the path to the dataset
    dataset_path = './datasets/my_voice/sentences.json'

    # Fine-tune the model
    SpeechRecognition.fine_tune_model(dataset_path)


if __name__ == "__main__":
    main()
