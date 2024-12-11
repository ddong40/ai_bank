import requests
from bs4 import BeautifulSoup
import os

# 이미지 저장할 폴더 생성
folder_name = 'images'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# 네이버 뉴스 검색 URL
search_url = 'https://search.naver.com/search.naver?where=news&query=중국어선'

# 검색 결과 페이지 요청
response = requests.get(search_url)
soup = BeautifulSoup(response.text, 'html.parser')

# 뉴스 링크와 제목 찾기
news_items = soup.find_all('a', {'href': True, 'data-clk': 'news'})

image_counter = 1  # 이미지 이름 카운터
max_images = 1000  # 최대 다운로드 이미지 수

for item in news_items:
    if image_counter > max_images:
        break  # 최대 이미지 수에 도달하면 종료

    news_url = item['href']
    news_title = item.get_text()

    # 제목에 '중국어선' 포함 여부 확인
    if '중국어선' in news_title:
        # 뉴스 페이지 요청
        news_response = requests.get(news_url)
        news_soup = BeautifulSoup(news_response.text, 'html.parser')

        # 이미지 URL 찾기
        img_meta = news_soup.find('meta', property='og:image')
        if img_meta:
            img_url = img_meta['content']

            # 이미지 다운로드
            try:
                img_data = requests.get(img_url)
                if img_data.status_code == 200:  # 요청 성공 확인
                    img_name = os.path.join(folder_name, f'chinese_boat_{image_counter}.jpg')
                    with open(img_name, 'wb') as img_file:
                        img_file.write(img_data.content)
                    print(f'다운로드 완료: {img_name}')
                    image_counter += 1  # 카운터 증가
                else:
                    print(f'이미지 다운로드 실패: {img_url}, 상태 코드: {img_data.status_code}')
            except Exception as e:
                print(f'다운로드 실패: {img_url}, 오류: {e}')
        else:
            print(f"뉴스 '{news_title}'에 이미지가 없습니다.")

print('이미지 다운로드 프로세스 완료!')
