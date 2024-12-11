import json
import csv
import os

# JSON 파일들이 있는 폴더 경로와 CSV 저장 경로를 지정합니다.

json_folder_path = "C:/Users/ddong40/Desktop/본프로젝트/AI Trainer/fitness_json/"
csv_path = "C:/Users/ddong40/Desktop/본프로젝트/AI Trainer/사이드런지_전체_1104.csv"

# 최종 CSV 파일에 저장할 데이터 리스트
csv_data = []

# 헤더 생성 (각 pts 좌표를 개별 열로 설정)
header = None

# 폴더 내 모든 JSON 파일 순회
for filename in os.listdir(json_folder_path):
    if filename.endswith(".json"):
        file_path = os.path.join(json_folder_path, filename)

        with open(file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)

            # "exercise"가 "푸시업"인 파일 필터링
            if data["type_info"]["exercise"] == "사이드 런지":
                # 첫 번째 파일에서 헤더 생성
                if header is None:
                    header = [f"{point}_{axis}" for point in data["frames"][0]["view5"]["pts"].keys() for axis in ["x", "y"]] + ['description']

                # 각 프레임의 좌표 정보와 조건값을 CSV 데이터로 추가
                for frame_index, frame in enumerate(data["frames"]):
                    for view, view_data in frame.items():
                        if view == "view5":  # 필요한 view만 사용
                            row = {}

                            # 좌표 값 추가
                            for point, coordinates in view_data["pts"].items():
                                row[f"{point}_x"] = coordinates["x"]
                                row[f"{point}_y"] = coordinates["y"]

                            # 운동 설명 추가
                            row['description'] = data["type_info"]['description']

                            # CSV 데이터 리스트에 추가
                            csv_data.append(row)

# CSV 파일 저장
with open(csv_path, 'w', newline='', encoding='cp949') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=header)
    writer.writeheader()  # 헤더 작성
    writer.writerows(csv_data)  # 데이터 작성

print(f"{csv_path} 파일이 생성되었습니다.")
