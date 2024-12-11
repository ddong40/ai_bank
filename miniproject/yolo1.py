
# yolofla


from flask import Flask, render_template, request, jsonify, Response
import cv2

app = Flask(__name__)

# 주차 정보 저장용 전역 변수
parking_info = {
    "car_count": 0,
    "empty_spots": 4
}

# 메인 페이지 라우트
@app.route('/')
def index():
    return render_template('flask.html', parking_info=parking_info)

# YOLO에서 데이터를 받는 라우트
@app.route('/update_parking_info', methods=['POST'])
def update_parking_info():
    global parking_info
    data = request.get_json()
    parking_info["car_count"] = data["car_count"]
    parking_info["empty_spots"] = data["empty_spots"]
    return jsonify({"message": "주차 정보가 업데이트되었습니다."})

# 비디오 피드를 스트리밍하는 라우트 추가
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# 비디오 프레임을 생성하는 함수 추가0
def gen_frames():
    cap = cv2.VideoCapture("http://192.168.0.109:5000/video_feed")  # CCTV 스트림 경로
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

if __name__ == "__main__":
    app.run(debug=True)
