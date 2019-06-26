from torchsummary import summary
from torchvision import models

from config import im_size

if __name__ == '__main__':
    model = models.segmentation.deeplabv3_resnet50(pretrained=False, progress=True, num_classes=20)
    summary(model, input_size=(3, im_size, im_size))
