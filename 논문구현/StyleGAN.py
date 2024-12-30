import torch
from torchvision import transforms
from PIL import Image

# 모델 로드 (paprika 스타일 사용)
model_path = "paprika.pt"
model = torch.hub.load("AK391/animegan2-pytorch:main", "generator")
model.load_state_dict(torch.load(model_path))
model.eval()

# 이미지 로드 및 전처리
def load_image(path):
    img = Image.open(path).convert("RGB")
    return img

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

img = load_image("C:/Users/noori/Pictures/사진/KakaoTalk_20241126_114702173.jpg")
img = transform(img).unsqueeze(0)

# 모델에 이미지 전달
with torch.no_grad():
    output = model(img)

# 후처리 및 저장
output = output.squeeze().permute(1, 2, 0).numpy()
output = (output * 0.5 + 0.5) * 255
output = output.astype("uint8")
output_img = Image.fromarray(output)
output_img.save("anime_style_image2_8.png")
