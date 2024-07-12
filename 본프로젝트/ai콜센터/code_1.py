# 라이브러리 불러오기
from langchain import LLMChain, PromptTemplate
from langchain.llms import OpenAI
import pandas as pd
import os
import re       # 정규 표현식
import speech_recognition as sr
from gtts import gTTS
import os
import playsound

# OpenAI API 키 설정

# speech to txt
def get_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source: 
        print("say something")
        audio = r.listen(source, timeout=10, phrase_time_limit=60)
        said = " "
        
        try:
            said = r.recognize_google(audio, language="ko-KR")
            print("your speech thinks like :", said)
        except Exception as e:
            print("Exception :", e)
    return said

# txt to speech
def speak(text):
    tts = gTTS(text=text, lang='ko')
    filename='voice.mp3'
    tts.save(filename)
    playsound.playsound(filename)
    os.remove(filename)
    
speak("안녕하세요, 누리다락입니다. 무엇을 도와드릴까요?")

# CSV 파일을 읽어서 답변 형식 템플릿을 로드하는 함수
def load_template_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    templates = {}
    for index, row in df.iterrows():        # iterrows : 행 반복 처리 
        templates[row['situation']] = {
            'template': row['response_template'],
            'keywords': row['keywords'].split('|')  # 키워드를 '|'로 구분해 저장
        }
    return templates

# 답변 형식을 저장한 CSV 파일 로드
response_templates = load_template_from_csv('C:/Users/ddong40/ai_2/본프로젝트/data/시나리오test1029 - 시트1.csv')

# 예약 데이터를 저장할 리스트와 테이블 정보 초기화
reservations = []  # 예약 정보가 축적될 리스트
table_info = {"2인석": 6, "4인석": 2}  # 2인석 6개, 4인석 2개
missing_info = []  # 누락된 정보를 순차적으로 처리하기 위한 리스트
reservation_info = {}  # 사용자 입력 정보를 저장할 딕셔너리

# 정보 추출 함수 (정규 표현식을 사용하여 예약 정보 추출)
def extract_reservation_info(user_input):
    global reservation_info
    date = re.search(r"\d{1,2}월 \d{1,2}일", user_input)
    time = re.search(r"\d{1,2}시 \d{1,2}분", user_input)
    people = re.search(r"\d+명", user_input)
    
    reservation_info = {
        "date": date.group() if date else None,
        "time": time.group() if time else None,
        "people": int(people.group()[:-1]) if people else None,
        "name": None,  # 예약자 이름은 수동으로 입력 받음
        "phone_num": None  # 휴대전화 뒷자리는 수동으로 입력 받음
    }

# 누락된 정보에 대한 질문을 생성하는 함수
def generate_question_for_missing_info(info_type):
    questions = {
        "date": "네. 예약 날짜는 언제로 하시겠습니까?",
        "time": "네. 예약 시간은 언제로 하시겠습니까?",
        "people": "네. 몇 분이 방문하실 예정인가요?",
        "name": "네. 예약자 성함이 어떻게 되세요?",
        "phone_num": "네. 휴대폰 뒷자리가 어떻게 되세요?"
    }
    return questions.get(info_type, "정보를 입력해 주세요.")

# 사용자 질문과 키워드를 비교해 알맞은 답변 템플릿 선택
def select_response_template(user_question):
    for situation, data in response_templates.items():
        # 각 상황별 키워드가 질문에 포함되는지 확인
        if any(keyword in user_question for keyword in data['keywords']):
            return data['template']
    # 기본 답변 (적절한 템플릿을 찾지 못한 경우)
    return "죄송합니다, 다시한번 말씀해주시겠어요?"

# 프롬프트 템플릿 설정
template = """
식당 이름: {restaurant_name}
테이블: {table_info}
운영 시간: {operating_hours}
예약 시간: {reservation_interval}
주차 가능 여부: {parking_availability}

질문: {user_question}

현재까지 예약 정보: {reservations}

위 정보를 바탕으로 상황에 맞는 형식으로 답변하세요:
{response_template}
"""

# 프롬프트 템플릿 초기화
prompt = PromptTemplate(
    input_variables=["restaurant_name", "table_info", "operating_hours", "reservation_interval", "parking_availability", "user_question", "reservations", "response_template"],
    template=template
)

# OpenAI LLM 설정
# llm = OpenAI(model="gpt-3.5-turbo")
llm = OpenAI(model="gpt-3.5-turbo-instruct")

# LLMChain 생성
chain = LLMChain(llm=llm, prompt=prompt)

# 대화형 챗봇
def chat(user_question):
    global missing_info, reservation_info, confirmation_stage, count

    # 예약 정보가 아직 비어있으면 사용자 질문에서 추출
    if not reservation_info:
        extract_reservation_info(user_question)
        missing_info = [key for key, value in reservation_info.items() if not value]

    # 누락된 정보가 있는 경우 하나씩 질문
    if missing_info:
        current_info_type = missing_info.pop(0)  # 첫 번째 누락된 정보를 가져옴
        follow_up_question = generate_question_for_missing_info(current_info_type)
        print("챗봇 응답:", follow_up_question)
        speak(follow_up_question)
        return current_info_type  # 현재 질문 유형을 반환해 다음 입력을 기다림

    # 모든 정보가 모였을 때 예약 완료 메시지 출력 및 추가 질문 단계로 전환
    if not missing_info and all(reservation_info.values()):
        if count == 0:
            print("챗봇 응답: 네, 예약되었습니다. 추가로 필요하신 사항 있으신가요?")
            confirmation_stage = True  # 추가 확인 단계로 설정
            count +=1
            return "confirmation"
        else:
            return "confirmation"

    # 모든 정보가 모였을 때만 응답 템플릿으로 답변 생성
    # response_template = select_response_template(user_question)

    # 사용자 질문과 예약 정보를 전달하여 응답 생성
    info = {
        "restaurant_name": "누리다락",
        "table_info": f"2인석 {table_info['2인석']}개, 4인석 {table_info['4인석']}개",
        "operating_hours": "11:00 ~ 21:00",
        "reservation_interval": "30분 단위",
        "parking_availability": "주차 불가",
        "user_question": user_question,
        "reservations": str(reservations),
        "response_template": response_template
    }
    
    # 응답 생성
    response = chain.run(info)
    print("챗봇 응답:", response)
    speak(response)

# 예약 정보 업데이트 함수
def update_reservation_info(info_type, user_response):
    if info_type == "date":
        reservation_info["date"] = re.findall(r"\d{1,2}월 \d{1,2}일", user_response)[0]
    elif info_type == "time":
        reservation_info["time"] = re.findall(r"\d{1,2}시", user_response)[0]
    elif info_type == "people":
        # reservation_info["people"] = int(user_response[0])
        reservation_info["people"] = user_response.split('명')
    elif info_type == "name":
        reservation_info["name"] = user_response[:3]
    elif info_type == "phone_num":
        reservation_info["phone_num"] = user_response[:4]

# 대화 예시
current_info_type = None
confirmation_stage = False  # 추가 정보 확인 단계
count = 0

while True:
    user_input = get_audio()
    if user_input.lower() == "exit":
        break
    
    # 추가 확인 단계에서 "아니요" 입력 시 종료
    if confirmation_stage:
        if user_input.lower() == "아니요":
            print("챗봇 응답: 감사합니다. 통화를 종료합니다.")
            speak('감사합니다. 통화를 종료합니다.')
            break
        elif '감사합니다' in user_input.lower():
            break
        elif user_input.lower():
            response_template = select_response_template(user_input)
            print("챗봇 응답:", response_template)
            speak(response_template)
        else:
            print("챗봇 응답: 추가로 요청하실 사항이 있으신가요?")
            speak('추가로 요청하실 사항이 있으신가요?')
            continue
    
    # 누락된 정보가 있는 경우, 그에 대한 응답을 업데이트
    if current_info_type:
        update_reservation_info(current_info_type, user_input)
        current_info_type = None
    
    # 필요한 정보를 모두 받은 경우 예약 완료 확인 메시지를 출력하고 추가 확인 단계로 진입
    current_info_type = chat(user_input)
    if current_info_type == "confirmation":
        confirmation_stage = True  # 추가 정보 확인 단계로 설정

print("======================")
print("최종 예약 정보:", reservation_info)
print("======================")
