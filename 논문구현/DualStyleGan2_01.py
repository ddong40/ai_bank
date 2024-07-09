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

class SingleStyleGAN:
    def __init__(self, gen_output_dim, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.generator = build_generator(100, gen_output_dim).to(self.device)

    def load_weights(self, weights_path):
        self.generator.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.generator.eval()

    def transform_image(self, image_path):
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
        stylized_image = self.generator(noise)

        return stylized_image.view(28, 28).detach().cpu()



# Example usage
if __name__ == "__main__":
    # Initialize SingleStyleGAN
    gen_output_dim = 784  # Assuming 28x28 images
    gan = SingleStyleGAN(gen_output_dim)

    # Load pre-trained weights
    gan.load_weights('C:/Users/ddong40/ai_2/generator.pt')

    # Transform input image
    input_image_path = "C:/Users/ddong40/Desktop/paper/image/me.jpg"
    output_image = gan.transform_image(input_image_path)

    # Save the output image
    output_image = (output_image + 1) / 2.0  # Denormalize
    output_image = transforms.ToPILImage()(output_image)
    output_image.save("C:/Users/ddong40/Desktop/paper/image/output_image.jpg")


