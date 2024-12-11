from transformers import GPT2LMHeadModel, GPT2Tokenizer

# GPT-2 모델과 토크나이저 불러오기
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# 모델을 평가(evaluation) 모드로 설정
model.eval()

def generate_text(context, max_length=150, temperature=0.7, top_k=50):
    """
    GPT-2를 사용해 텍스트를 생성하는 함수.

    Args:
        context (str): 입력 텍스트 (프롬프트).
        max_length (int): 생성될 텍스트의 최대 길이.
        temperature (float): 생성의 랜덤성을 조절.
        top_k (int): 상위 k개의 단어만 샘플링에 사용.

    Returns:
        str: 생성된 텍스트.
    """
    # 입력 텍스트를 토큰화
    input_ids = tokenizer.encode(context, return_tensors="pt")

    # 텍스트 생성
    output = model.generate(
        input_ids,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        do_sample=True,  # 샘플링 활성화
        pad_token_id=tokenizer.eos_token_id  # 문장이 끝났음을 알려주는 토큰
    )

    # 생성된 토큰을 텍스트로 디코딩
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# 예시 사용
context = """Today, I wake up early as usual and start the same day.
But it snowed so much that the roads were frozen solid.
         """
generated = generate_text(context, max_length=200)
print("Generated Text:\n", generated)
