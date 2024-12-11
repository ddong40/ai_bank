from transformers import pipeline

# 1. 감정 분석 파이프라인 생성
classifier = pipeline("sentiment-analysis")

# 2. 텍스트 입력
texts = [
    "I love this product! It's amazing and works perfectly.",
    "This is the worst experience I've ever had. Completely disappointed."
]

# 3. 감정 분석 수행
results = classifier(texts)

# 4. 결과 출력
for text, result in zip(texts, results):
    print(f"텍스트: {text}")
    print(f"분석 결과: {result}")
    print("-" * 50)
    



