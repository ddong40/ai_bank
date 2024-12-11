import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urlparse

# 웹 페이지 URL
url = 'https://www.seoul.co.kr/news/society/accident/2024/10/05/20241005500037?wlog_tag3=naver'

# 페이지 요청
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# 이미지 저장할 폴더 생성
folder_name = 'images'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# 이미지 URL 찾기
img_meta = soup.find('meta', property='og:image')
if img_meta:
    img_url = img_meta['content']

    # URL에서 파일 이름 추출 (쿼리 문자열 제거)
    parsed_url = urlparse(img_url)
    img_name = os.path.join(folder_name, os.path.basename(parsed_url.path))

    # 이미지 다운로드
    try:
        img_data = requests.get(img_url).content
        with open(img_name, 'wb') as img_file:
            img_file.write(img_data)
        print(f'다운로드 완료: {img_name}')
    except Exception as e:
        print(f'다운로드 실패: {img_url}, 오류: {e}')
else:
    print("이미지가 없습니다.")

print('이미지 다운로드 프로세스 완료!')
