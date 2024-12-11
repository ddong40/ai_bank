import supervision as sv
import numpy as np
import cv2
from sam2.build_sam import build_sam2_video_predictor

# 경로 설정
PATH = './논문/sam2'
VIDEO_PATH = 'C:/Users/noori/Pictures/영상/단추3.mp4'
SEGMENTED_VIDEO_PATH = PATH + '/notebooks/videos/단추3.mp4'

CHECKPOINT = PATH + "/checkpoints/sam2.1_hiera_large.pt"
CONFIG = "C:/ai5/논문/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"

# SAM2 모델 로드
sam2_model = build_sam2_video_predictor(CONFIG, CHECKPOINT)

# 비디오 세그멘테이션
def process_video_with_sam2(video_path, output_video_path, sam2_model):
    cap = cv2.VideoCapture(video_path)
    video_info = sv.VideoInfo.from_video_path(video_path)

    # SAM2 모델 초기화
    inference_state = sam2_model.init_state(video_path)
    sam2_model.reset_state(inference_state)

    colors = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700']
    mask_annotator = sv.MaskAnnotator(
        color=sv.ColorPalette.from_hex(colors),
        color_lookup=sv.ColorLookup.TRACK
    )

    # 초기 포인트 설정 (첫 번째 프레임)
    ret, initial_frame = cap.read()
    if not ret:
        print("비디오를 읽을 수 없습니다.")
        return

    # 초기 객체 포인트와 레이블 설정
    # 예제: 두 개의 포인트를 지정
    initial_points = np.array([[590, 950], [620, 1100]], dtype=np.float32)
    labels = np.array([1, 1])  # 객체의 클래스 라벨 (1은 포그라운드)

    frame_idx = 0
    tracker_id = 1

    _, object_ids, mask_logits = sam2_model.add_new_points(
        inference_state=inference_state,
        frame_idx=frame_idx,
        obj_id=tracker_id,
        points=initial_points,
        labels=labels,
    )

    # 세그멘테이션 처리
    with sv.VideoSink(output_video_path, video_info=video_info) as sink:
        for frame_idx, object_ids, mask_logits in sam2_model.propagate_in_video(inference_state):
            ret, frame = cap.read()
            if not ret:
                break

            # 마스크 처리
            masks = (mask_logits > 0.0).cpu().numpy()
            N, X, H, W = masks.shape
            masks = masks.reshape(N * X, H, W)

            detections = sv.Detections(
                xyxy=sv.mask_to_xyxy(masks=masks),
                mask=masks,
                tracker_id=np.array(object_ids)
            )

            # 마스크를 프레임에 적용
            frame = mask_annotator.annotate(frame, detections)
            sink.write_frame(frame)

    cap.release()
    print(f"세그멘테이션된 비디오가 {output_video_path}에 저장되었습니다.")

# 실행
process_video_with_sam2(VIDEO_PATH, SEGMENTED_VIDEO_PATH, sam2_model)