import warnings
warnings.filterwarnings('ignore')
from torchvision import models
import numpy as np
import cv2
import requests
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from PIL import Image
from efficientnet_pytorch import EfficientNet





if __name__ == "__main__":
    # model = models.resnet50(pretrained=True)
    # model.layer4[0]


    model = EfficientNet.from_pretrained('efficientnet-b4')
    # model._fc = nn.Linear(1792, len(classes)) 


    # model = models.resnet50(pretrained=True)
    model.eval()
    image_url = "/kaggle/input/glaucoma-dataset-eyepacs-airogs-light-v2/eyepac-light-v2-512-jpg/train/RG/EyePACS-DEV-RG-1.jpg"
    img = np.array(Image.open(image_url))
    img = cv2.resize(img, (224, 224))
    img = np.float32(img) / 255
    input_tensor = preprocess_image(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


    # The target for the CAM is the Bear category.
    # As usual for classication, the target is the logit output
    # before softmax, for that category.
    targets = [ClassifierOutputTarget(295)]
    target_layers = [model._blocks[31]] #[model.layer4]
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cams = cam(input_tensor=input_tensor, targets=targets)
        cam_image = show_cam_on_image(img, grayscale_cams[0, :], use_rgb=True)
    cam = np.uint8(255*grayscale_cams[0, :])
    cam = cv2.merge([cam, cam, cam])
    images = np.hstack((np.uint8(255*img), cam , cam_image))
    Image.fromarray(images)
