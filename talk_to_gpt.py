import speech_recognition as sr
import requests
import json


# Function to send message to ChatGPT
def send_to_chatgpt(message):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer YOUR_OPENAI_API_KEY"
    }
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message}
        ]
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    return response.json()


# Function to filter text for specific words or phrases
def filter_text(text, word):
    if word in text:
        return True
    return False


# Function to continuously record and process audio
def record_and_process():
    r = sr.Recognizer()
    pre_defined_string = ""
    while True:
        with sr.Microphone() as source:
            print("Listening...")
            audio = r.listen(source)
            try:
                text = r.recognize_google(audio)
                print("You said: {}".format(text))
                if filter_text(text, "send"):
                    text = text.rsplit(' ', 1)[0]  # Remove the last word "send"
                    pre_defined_string += text
                    response = send_to_chatgpt(pre_defined_string)
                    print("ChatGPT Response: ", response)
                    pre_defined_string = ""  # Reset the string after sending
            except sr.UnknownValueError:
                print("Google Speech Recognition could not understand audio")
            except sr.RequestError as e:
                print("Could not request results from Google Speech Recognition service; {0}".format(e))


# Start the recording and processing
# record_and_process()


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arg', help="description", default=False, const=True, nargs='?')

    return parser.parse_args()


def main():
    args = parse()
    pass


if __name__ == '__main__':
    main()
