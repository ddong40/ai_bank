import requests
from bs4 import BeautifulSoup
import os

# 웹 페이지 URL (뉴스 사이트의 메인 페이지를 사용하세요)
url = 'https://example.com/news'  # 크롤링할 뉴스 페이지 URL로 변경하세요.

# 페이지 요청
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# 이미지 저장할 폴더 생성
folder_name = '어선_이미지'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# 뉴스 기사 찾기
articles = soup.find_all('article')  # 기사 태그에 맞게 수정하세요

for article in articles:
    title = article.find('h2')  # 제목 태그에 맞게 수정하세요
    if title and '중국어선' in title.text:
        # 이미지 찾기
        img = article.find('img')  # 이미지 태그에 맞게 수정하세요
        if img:
            img_url = img.get('src')
            
            # 이미지 URL이 절대 경로가 아닐 경우 절대 경로로 변환
            if not img_url.startswith('http'):
                img_url = requests.compat.urljoin(url, img_url)

            # 이미지 파일 이름 생성
            img_name = os.path.join(folder_name, img_url.split('/')[-1])
            
            # 이미지 다운로드
            img_data = requests.get(img_url).content
            with open(img_name, 'wb') as img_file:
                img_file.write(img_data)

print('이미지 다운로드 완료!')