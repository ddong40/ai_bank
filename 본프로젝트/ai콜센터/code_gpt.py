import openai
import speech_recognition as sr
from gtts import gTTS
import os
import playsound

# OpenAI API 키 설정

def speak(text):
    tts = gTTS(text=text, lang='ko')
    filename = 'voice.mp3'
    tts.save(filename)
    playsound.playsound(filename)
    os.remove(filename)

# 이전 대화 내용을 저장하는 리스트를 전역 변수로 설정
messages = [
    {"role": "system", "content": """당신은 누리다락 식당 예약을 도와주는 챗봇입니다. 다음 정보에 기반하여 사용자 질문에 친절하게 답변해야 합니다. 
        예약과 관련하여 사용자가 답변한 내용에 오류가 없으면 다시 질문하지 마세요. :

가게 이름: 누리다락,
운영 시간: 매일 오전 11시부터 오후 9시까지,
예약 가능 시간: 오전 11시반, 오후12시, 오후12시반, 오후1시, 오후1시반, 오후2시, 오후2시반, 오후 5시반, 오후6시, 오후6시반, 오후7시, 오후7시반,
브레이크타임 : 오후 3시-오후5시,
좌석 : 2인석 6개, 4인석 2개, 
메뉴 : 크림파스타, 로제파스타, 토마토파스타, 게살크림리조또, 목살 필라프, 새우 필라프, 토마호크 스테이크(350g 1인), 콜라, 제로콜라, 사이다, 환타, 자몽에이드, 청포도에이도, 레몬에이드,
주차: 주변 공영주차장 이용,
유아가 있다고 하면 유아용 의자가 필요한지 문의,
예약 절차: 예약날짜 -> 예약시간 -> 인원 수 -> 이름 -> 휴대폰 뒷번호 4자리
예약인원이 최대 수용가능 테이블인 4인석보다 많을 경우 예약 불가.
예약 날짜가 오늘보다 이전일일 경우에는 예약이 불가합니다. 
당일 예약 불가.
연중무휴.
예약 내용 확인 후 사용자에게 예약 내용이 맞는지 질문한 후 사용자가 맞다고 대답하면 예약 완료. 
예약이 완료되면 사용자가 응답한 내용 요약 후 예약이 완료되었습니다 출력, 그 후 사전에 메뉴 예약할지 문의.
메뉴 예약 희망시, 메뉴 기반으로 하여 메뉴 예약.
메뉴 예약내용이 맞는지 질문 후 메뉴 예약 완료.
메뉴 예약 완료 후 추가로 도움이 필요한 것이 있는지 문의.
추가로 도움이 필요한 내용이 없으면 통화를 종료합니다. 출력하기
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

# 대화 루프 시작
# def start_conversation():
#     speak("안녕하세요, 누리다락입니다. 무엇을 도와드릴까요?")

#     while True:
#         user_input = get_audio().lower()
        
#         if "종료" in user_input or "감사합니다" in user_input:
#             speak("통화를 종료합니다. 감사합니다.")
#             break
#         else:
#             bot_response = get_chatbot_response(user_input)
#             print("챗봇 응답:", bot_response)
#             speak(bot_response)

def start_conversation():
    # speak("안녕하세요, 누리다락입니다. 무엇을 도와드릴까요?")

    while True:
        # user_input = get_audio().lower()
        user_input = input('입력 : ')
        
        if "종료" in user_input or "감사합니다" in user_input:
            speak("통화를 종료합니다. 감사합니다.")
            break
        else:
            bot_response = get_chatbot_response(user_input)
            print("챗봇 응답:", bot_response)
            # speak(bot_response)

# 대화 시작
start_conversation()
