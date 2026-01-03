import speech_recognition as sr
import pyautogui

r = sr.Recognizer()

# List all microphones
mics = sr.Microphone.list_microphone_names()
for i, mic in enumerate(mics):
    print(f'{i}: {mic}')

# Choose the microphone (replace 0 with the index of your microphone)
mic_index = 0

# Mapping from Hebrew characters to English characters on the keyboard
hebrew_to_english = {
    'א': 't',
    'ב': 'c',
    'ג': 'd',
    'ד': 's',
    'ה': 'v',
    'ו': 'u',
    'ז': 'z',
    'ח': 'j',
    'ט': 'y',
    'י': 'h',
    'כ': 'f',
    'ל': 'k',
    'מ': 'n',
    'נ': 'b',
    'ס': 'x',
    'ע': 'g',
    'פ': 'p',
    'צ': 'm',
    'ק': 'e',
    'ר': 'r',
    'ש': 'a',
    'ת': ',',
    'ם': 'o',
    'ן': 'i',
    'ף': ';',
    'ץ': '.',
    ' ': ' ',
    '.': '.',
    ',': ',',
    '?': '?',
    '!': '!'
    # Add more characters if needed
}

while True:
    with sr.Microphone(device_index=mic_index) as source:
        # Adjust for ambient noise
        r.adjust_for_ambient_noise(source)
        print('Listening...')
        try:
            # Listen with a timeout
            audio = r.listen(source, timeout=5)
            # Save audio to a file
            with open('audio.wav', 'wb') as f:
                f.write(audio.get_wav_data())
            text = r.recognize_google(audio, language='he')
            print('You said: {}'.format(text))
            # Convert Hebrew text to English characters
            english_text = ''.join(hebrew_to_english.get(char, char) for char in text)
            pyautogui.write(english_text)
        except sr.WaitTimeoutError:
            continue
        except Exception as e:
            print('An error occurred: {}'.format(str(e)))
