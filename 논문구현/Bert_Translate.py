## tf274gpu ##

from transformers import MarianMTModel, MarianTokenizer
import warnings
warnings.filterwarnings('ignore')

# 번역 모델과 토크나이저 설정 (English → French)
model_name = "Helsinki-NLP/opus-mt-en-fr"  # English to French
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate_text(text, tokenizer, model):
    """텍스트를 번역하는 함수"""
    # 입력 텍스트를 토큰화
    tokenized_text = tokenizer.prepare_seq2seq_batch([text], return_tensors="pt", padding=True)
    # 모델로 번역 수행
    translated = model.generate(**tokenized_text)
    # 번역 결과 디코딩
    translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
    return translated_text

# 영어 문장 입력
english_text = "This re-release, titled The Next Day Extra, was presented in the form of three disks: the original album, \
    unpublished studio sessions and remixes, plus a DVD containing the four clips that have already been unveiled."

# 영어 → 프랑스어 번역
french_translation = translate_text(english_text, tokenizer, model)
print("French Translation:", french_translation)

# 프랑스어 → 영어 모델 로드 (French → English)
model_name_fr_en = "Helsinki-NLP/opus-mt-fr-en"  # French to English
tokenizer_fr_en = MarianTokenizer.from_pretrained(model_name_fr_en)
model_fr_en = MarianMTModel.from_pretrained(model_name_fr_en)

# 프랑스어 입력
french_text = "Un homme expliquait que le fonctionnement de la hernia fonctionnelle qu'il avait reconnu avant de faire, \
    le fonctionnement de la hernia fonctionnelle que j'ai réussi, j'ai réussi."

# 프랑스어 → 영어 번역
english_translation = translate_text(french_text, tokenizer_fr_en, model_fr_en)
print("English Translation:", english_translation)
