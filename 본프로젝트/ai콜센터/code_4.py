import openai
import speech_recognition as sr
from gtts import gTTS
import os
import playsound
# import pyttsx3
from datetime import datetime, timedelta
import requests
import json

# OpenAI API 키 설정

# Typecast api
TYPECAST_API_KEY = '__plt9onJuVApvTfRbQH4dufxRQAk8wmsKx7vt9eUpcME'  # 여기에 Typecast API 키를 입력하세요.
TYPECAST_API_URL = 'https://typecast.ai/api/speak'  # Typecast API URL


def speak(text):
    headers = {
        'Authorization': f'Bearer {TYPECAST_API_KEY}',
        'Content-Type': 'application/json',
    }
    
    # TTS 요청 생성
    data = {
        "text": text,
        "lang": "auto",  # 원하는 목소리 옵션을 입력하세요. (예: "ko-KR-Gyeonggi")
        "tempo": 1.0,
        "pitch": 1.0,
        "volume": 1.0,
        "actor_id" : "661797923ed12f31b61c4b5f"
        }
    
    response = requests.post(TYPECAST_API_URL, headers=headers, json = data)

    if response.status_code == 200:
        # 성공적으로 음성 파일을 생성한 경우
        audio_url = response.json().get('audio_url')  # 음성 URL 추출
        playsound.playsound(audio_url)  # 음성 재생
    else:
        print("TTS 오류 발생:", response.json())



# 날짜별 예약 가능한 좌석 초기화 함수
def initialize_slots():
    times = ["오후 12시", "오후 1시", "오후 2시", "오후 5시", "오후 6시", "오후 7시", "오후 8시", "오후 9시"]
    today = datetime.now().date()
    reservation_slots = {}

    # 14일 간의 예약 데이터를 초기화합니다.
    for i in range(14):
        date = today.strftime("%Y-%m-%d")
        reservation_slots[date] = {time: 8 for time in times}  # 좌석 초기화
        today += timedelta(days=1)
    
    return reservation_slots

# 날짜 및 시간별 예약 가능한 좌석 관리
reservation_slots = initialize_slots()

# 이전 대화 내용을 저장하는 리스트를 전역 변수로 설정
messages = [
    {"role": "system", "content": """당신은 누리카세 식당 예약을 도와주는 챗봇입니다. 다음 정보에 기반하여 사용자 질문에 존댓말로 간결하고 친절하게 답변해야 합니다. 
        예약과 관련하여 사용자가 답변한 내용에 오류가 없으면 다시 질문하지 마세요.:

가게 이름: 누리카세,
운영 시간: 매일 오후 12시부터 오후 10시까지,
예약 가능 시간: 오후 12시, 오후 1시, 오후 2시, 오후 5시, 오후 6시, 오후 7시, 오후 8시, 오후 9시 
브레이크타임 : 오후 3시 - 오후 5시,
좌석 : 8석,
메뉴 : 런치오마카세(5만원), 디너오마카세(8만원)
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
    global messages, reservation_slots  # 전역 변수로 선언된 messages, reservation_slots 사용

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

    # 예약 관련 응답에서 예약 날짜와 시간을 확인하고 좌석 차감
    if "예약 완료" in bot_response:
        # 예약 날짜와 시간 및 인원 확인
        date = "예약 날짜 추출 로직"  # 예약 날짜 추출 (예: YYYY-MM-DD 형식)
        time_slot = "예약 시간대 추출 로직"  # 예약 시간대 추출 (예: 오후 1시)
        num_people = int(user_input.split("명")[0].split()[-1])  # 인원수 추출

        # 좌석 차감 및 남은 좌석 확인
        if reservation_slots.get(date) and reservation_slots[date].get(time_slot):
            if reservation_slots[date][time_slot] >= num_people > 0:
                reservation_slots[date][time_slot] -= num_people
                bot_response += f"\n남은 좌석: {reservation_slots[date][time_slot]}석"
            else:
                bot_response = f"{date} {time_slot}에 남은 좌석이 부족합니다. 다른 시간대를 선택해주세요."
        else:
            bot_response = f"{date}에 예약이 불가능합니다. 다른 날짜를 선택해주세요."

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
    speak("안녕하세요, 누리다락입니다. 무엇을 도와드릴까요?")

    while True:
        user_input = input('입력 : ')
        
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
