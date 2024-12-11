from ultralytics import SAM
import cv2
import numpy as np

model = SAM('sam_b.pt')
image_path = 'segment-anything-main/segment-anything-main/notebooks/images/dog.jpg'

# 바운딩 박스를 사용한 세그멘테이션
results_bbox = model(image_path, bboxes=[100, 150, 200, 300])

# 단일 점을 사용한 세그멘테이션
results_points = model(image_path, points=[250, 300], labels=[1])

# 여러 점을 사용한 세그멘테이션
results_multiple_points = model(image_path, points=[[100, 150], [200, 250]], labels=[1, 1])


# 결과 마스크 저장 (결과는 이진 마스크로 제공됨)
cv2.imwrite('output_mask_bbox.jpg', results_bbox[0].mask.astype(np.uint8) * 255)
# cv2.imwrite('output_mask_points.jpg', results_points[0].mask.astype(np.uint8) * 255)

mask_image_path = 'output_mask_bbox.jpg'

mask = cv2.imread(mask_image_path, cv2.COLOR_RGB2BGR)

cv2.imshow('Mask Image', mask)
cv2.waitKey(0)  # 아무 키나 누르면 창 닫기
cv2.destroyAllWindows()