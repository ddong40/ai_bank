import openai
import speech_recognition as sr
from gtts import gTTS
import os
import playsound
import pyttsx3

# OpenAI API 키 설정

def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 195)
    engine.say(text)
    engine.runAndWait()


# 이전 대화 내용을 저장하는 리스트를 전역 변수로 설정
messages = [
    {"role": "system", "content": """당신은 누리카세 식당 예약을 도와주는 챗봇입니다. 다음 정보에 기반하여 사용자 질문에 존댓말로 간결하고 친절하게 답변해야 합니다. 
        예약과 관련하여 사용자가 답변한 내용에 오류가 없으면 다시 질문하지 마세요.:

가게 이름: 누리카세,
운영 시간: 매일 오후 12시부터 오후 10시까지,
예약 가능 시간: 오후 12시, 오후 1시, 오후 2시, 오후 5시, 오후 6시, 오후 7시, 오후 8시, 오후 9시 
브레이크타임 : 오후 3시 - 오후 5시,
좌석 : 8석,
메뉴 : 런치오마카세, 디너오마카세
주차: 주변 공영주차장 이용,
유아가 있다고 하면 유아용 의자가 필요한지 문의,
예약 절차: 예약날짜 -> 예약시간 -> 인원 수 -> 이름 -> 휴대폰 뒷번호 4자리
예약인원 최대 8명까지 가능.
만약 예약 날짜가 오늘 날짜 기준 이전 일인 경우에는 예약이 불가.
만약 사용자가 오늘 날짜 이전일을 예약할 경우 다른 날짜로 예약 안내. 
당일 예약 불가.
연중무휴.
예약 내용 확인 후 사용자에게 예약 내용이 맞는지 질문한 후 사용자가 맞다고 대답하면 예약 완료. 
예약이 완료되면 예약이 완료되었습니다 출력.
알러지여부에 대해 문의. 
알러지가 있을 경우 예약 정보에 추가.
추가로 도움이 필요한 것이 있는지 문의.
추가로 도움이 필요한 내용이 없으면 통화가 종료됩니다. 출력하기.
    """}
]

def get_chatbot_response(user_input):
    global messages  # 전역 변수로 선언된 messages 사용
    # 사용자 메시지를 messages에 추가
    messages.append({"role": "user", "content": user_input})
    
    # OpenAI API를 호출하여 응답 생성
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=200,
        temperature=0.3
    )
    
    # 챗봇의 응답을 messages에 추가하여 대화 맥락 유지
    bot_response = response['choices'][0]['message']['content'].strip()
    messages.append({"role": "assistant", "content": bot_response})

    return bot_response

def get_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("말씀하세요...")
        audio = r.listen(source, timeout=10, phrase_time_limit=60)
        said = ""
        
        try:
            said = r.recognize_google(audio, language="ko-KR")
            print("인식된 내용:", said)
        except Exception as e:
            print("오류 발생:", e)
    return said

def start_conversation():
    speak("안녕하세요, 누리카세입니다. 무엇을 도와드릴까요?")

    while True:
        user_input = get_audio().lower()
        # user_input = input('입력 : ')
        
        if "종료" in user_input or "감사합니다" in user_input:
            print("통화를 종료합니다. 감사합니다.")
            speak("통화를 종료합니다. 감사합니다.")
            break
        else:
            bot_response = get_chatbot_response(user_input)
            print("챗봇 응답:", bot_response)
            speak(bot_response)

# 대화 시작
start_conversation()
