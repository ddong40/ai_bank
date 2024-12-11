from PIL import Image
import os

def split_image(image_path, tile_size, output_dir):
    """
    이미지를 일정한 크기의 타일로 분할하는 함수

    :param image_path: 분할할 이미지 파일의 경로
    :param tile_size: 타일 크기 (가로, 세로 크기 튜플 형식)
    :param output_dir: 분할된 타일을 저장할 폴더 경로
    """
    # 이미지 열기
    img = Image.open(image_path)
    img_width, img_height = img.size

    # 타일을 저장할 폴더가 없으면 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 타일 크기 지정
    tile_width, tile_height = tile_size

    # 이미지 크기를 타일 크기로 나누어 타일 개수 계산
    x_tiles = img_width // tile_width
    y_tiles = img_height // tile_height

    # 타일 분할 시작
    for x in range(x_tiles + 1):
        for y in range(y_tiles + 1):
            # 각 타일의 좌상단과 우하단 좌표 계산
            left = x * tile_width
            upper = y * tile_height
            right = min(left + tile_width, img_width)  # 이미지 경계 넘지 않도록
            lower = min(upper + tile_height, img_height)

            # 이미지에서 타일 크기만큼 크롭
            tile = img.crop((left, upper, right, lower))

            # 타일 파일 이름 생성
            tile_name = f'tile_{x}_{y}.png'

            # 타일 저장
            tile.save(os.path.join(output_dir, tile_name))

    print(f"이미지가 {tile_width}x{tile_height} 크기의 타일로 분할되었습니다.")

# 예시: 10980x10980 이미지를 1024x1024 타일로 분할

image_path = 'C:/Users/ddong40/ai_2/factory/data/샘플데이터.png'  # 분할할 이미지 경로
output_dir = 'C:/Users/ddong40/ai_2/factory/data/209/'  # 타일을 저장할 폴더
tile_size = (209, 209)  # 타일 크기 지정 (1024x1024)

split_image(image_path, tile_size, output_dir)
