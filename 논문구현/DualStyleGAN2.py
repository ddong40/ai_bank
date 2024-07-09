import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Generator network
def build_generator(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, output_dim),
        nn.Tanh()
    )

class DualStyleGAN:
    def __init__(self, gen_output_dim, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.generator_style1 = build_generator(100, gen_output_dim).to(self.device)
        self.generator_style2 = build_generator(100, gen_output_dim).to(self.device)

    def load_weights(self, style1_path, style2_path):
        self.generator_style1.load_state_dict(torch.load(style1_path, map_location=self.device))
        self.generator_style2.load_state_dict(torch.load(style2_path, map_location=self.device))
        self.generator_style1.eval()
        self.generator_style2.eval()

    def transform_image(self, image_path, style="style1"):
        # Load and preprocess image
        transform = transforms.Compose([
            transforms.Resize((28, 28)),  # Assuming 28x28 resolution
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        image = Image.open(image_path).convert("L")
        image_tensor = transform(image).view(1, -1).to(self.device)

        # Generate stylized image
        noise = torch.randn(1, 100, device=self.device)
        if style == "style1":
            stylized_image = self.generator_style1(noise)
        elif style == "style2":
            stylized_image = self.generator_style2(noise)
        else:
            raise ValueError("Invalid style. Choose 'style1' or 'style2'.")

        return stylized_image.view(28, 28).detach().cpu()

# Example usage
if __name__ == "__main__":
    # Initialize DualStyleGAN
    gen_output_dim = 784  # Assuming 28x28 images
    gan = DualStyleGAN(gen_output_dim)

    # Load pre-trained weights
    gan.load_weights("style1_weights.pth", "style2_weights.pth")

    # Transform input image
    input_image_path = "input_image.jpg" # 이미지 input
    output_style = "style1"  # Choose between 'style1' and 'style2' #스타일 설정
    output_image = gan.transform_image(input_image_path, style=output_style)

    # Save the output image
    output_image = (output_image + 1) / 2.0  # Denormalize
    output_image = transforms.ToPILImage()(output_image)
    output_image.save("output_image.jpg") #아웃풋 이미지 이름 설정 및 저장

