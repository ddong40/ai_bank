from transformers import pipeline

# Pre-trained 모델 불러오기 (질문-응답을 위한 모델)
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# 본문과 질문 설정
context = """
Tom goes everywhere with Catherine Green, a 54-year-old secretary. He moves around her office at work and goes shopping with her. 
"Most people don’t seem to mind Tom," says Catherine, who thinks he is wonderful. "He’s my fourth child," she says. 
She may think of him and treat him that way as her son. He moves around buying his food, paying his health bills and his taxes, 
but in fact Tom is a dog.

Catherine and Tom live in Sweden, a country where everyone is expected to lead an orderly life according to rules laid down by the government, 
which also provides a high level of care for its people. This level of care costs money.
"""

questions = [
    "How old is Catherine?",
    "Where does she live?",
    "Who is Tom?",
    "What country does Catherine live in?"
]

# 각 질문에 대해 답변 생성
for question in questions:
    result = qa_pipeline(question=question, context=context)
    print(f"Q: {question}")
    print(f"A: {result['answer']}\n")




