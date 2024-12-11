import openai
import speech_recognition as sr
from gtts import gTTS
import os
import playsound


import openai
import speech_recognition as sr
from gtts import gTTS
import os
import playsound

def select_language(prompt):
    languages = {
        "1": ("ko", "한국어"),
        "2": ("en", "영어"),
        "3": ("ja", "일본어"),
        "4": ("zh", "중국어"),
        "5": ("de", "독일어"),
        "6": ("es", "스페인어")
    }

    print(prompt)
    for key, (code, name) in languages.items():
        print(f"{key}: {name}")

    choice = input("선택: ")
    return languages.get(choice, ("en", "영어"))  # 기본값은 영어

def transcribe_speech_to_text(input_lang_code):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something...")
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio, language=input_lang_code)
            print(f"Recognized: {text} (Input language: {input_lang_code})")
            return text
        except sr.UnknownValueError:
            print("Could not understand audio.")
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
        return None

def translate_text_with_openai(text, target_language="en"):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    prompt = f"Translate the following text to {target_language}:\n{text}"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful translation assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100
        )
        translation = response['choices'][0]['message']['content'].strip()
        print(f"Translated: {translation}")
        return translation
    except Exception as e:
        print(f"Error in OpenAI API call: {e}")
        return None

def text_to_speech(text, lang="en"):
    try:
        tts = gTTS(text=text, lang=lang)
        filename = "output.mp3"
        tts.save(filename)
        playsound.playsound(filename)
        os.remove(filename)
    except Exception as e:
        print(f"Error in TTS: {e}")

def main():
    print("Starting simultaneous interpretation...")
    input_lang_code, input_lang_name = select_language("입력 언어를 선택하세요:")
    target_lang_code, target_lang_name = select_language("번역할 언어를 선택하세요:")

    print(f"Selected input language: {input_lang_name}")
    print(f"Selected target language: {target_lang_name}")

    while True:
        input_text = transcribe_speech_to_text(input_lang_code)
        if input_text:
            translated_text = translate_text_with_openai(input_text, target_language=target_lang_code)
            if translated_text:
                text_to_speech(translated_text, lang=target_lang_code)

if __name__ == "__main__":
    main()