import streamlit as st
import openai
import speech_recognition as sr
from gtts import gTTS
import os
import playsound

def select_language_ui():
    languages = {
        "한국어": "ko",
        "영어": "en",
        "일본어": "ja",
        "중국어": "zh",
        "독일어": "de",
        "스페인어": "es"
    }
    input_lang_name = st.selectbox("입력 언어를 선택하세요:", options=list(languages.keys()))
    target_lang_name = st.selectbox("번역 언어를 선택하세요:", options=list(languages.keys()))

    return languages[input_lang_name], languages[target_lang_name], input_lang_name, target_lang_name

def transcribe_speech_to_text(input_lang_code):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("음성을 입력하려면 버튼을 클릭하세요.")
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio, language=input_lang_code)
            return text
        except sr.UnknownValueError:
            st.error("음성을 인식하지 못했습니다. 다시 시도해주세요.")
        except sr.RequestError as e:
            st.error(f"Google Speech Recognition 서비스에 문제가 발생했습니다: {e}")
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
        return translation
    except Exception as e:
        st.error(f"번역 중 오류가 발생했습니다: {e}")
        return None

def text_to_speech(text, lang="en"):
    try:
        tts = gTTS(text=text, lang=lang)
        filename = "output.mp3"
        tts.save(filename)
        playsound.playsound(filename)
        os.remove(filename)
    except Exception as e:
        st.error(f"TTS 생성 중 오류가 발생했습니다: {e}")

def main():
    st.title("동시통역 프로그램")

    with st.sidebar:
        st.header("언어 설정")
        input_lang_code, target_lang_code, input_lang_name, target_lang_name = select_language_ui()

    st.write(f"### 선택된 입력 언어: {input_lang_name} ({input_lang_code})")
    st.write(f"### 선택된 번역 언어: {target_lang_name} ({target_lang_code})")

    if st.button("음성 입력 시작"):
        input_text = transcribe_speech_to_text(input_lang_code)
        if input_text:
            st.write(f"**입력된 텍스트:** {input_text}")
            translated_text = translate_text_with_openai(input_text, target_language=target_lang_code)
            if translated_text:
                st.write(f"**번역된 텍스트:** {translated_text}")
                if st.button("번역된 텍스트 음성 출력"):
                    text_to_speech(translated_text, lang=target_lang_code)

if __name__ == "__main__":
    main()
