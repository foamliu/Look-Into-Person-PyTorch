# import the necessary packages
import os
import random

import cv2 as cv
import numpy as np
import torch
from torchvision import transforms

from config import device, im_size, num_classes
from data_gen import random_choice, safe_crop, to_bgr, data_transforms
from utils import ensure_folder

if __name__ == '__main__':
    checkpoint = 'BEST_checkpoint.tar'
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']
    model = model.to(device)
    model.eval()

    transformer = data_transforms['valid']

    ensure_folder('images')

    test_images_folder = 'data/instance-level_human_parsing/Testing/Images'
    id_file = 'data/instance-level_human_parsing/Testing/test_id.txt'
    with open(id_file, 'r') as f:
        names = f.read().splitlines()

    samples = random.sample(names, 10)

    for i in range(len(samples)):
        image_name = samples[i]
        filename = os.path.join(test_images_folder, image_name + '.jpg')
        image = cv.imread(filename)
        image_size = image.shape[:2]

        x, y = random_choice(image_size)
        image = safe_crop(image, x, y)
        print('Start processing image: {}'.format(filename))

        x_test = torch.zeros((1, 3, im_size, im_size), dtype=torch.float)
        img = image[..., ::-1]  # RGB
        img = transforms.ToPILImage()(img)
        img = transformer(img)
        x_test[0:, 0:3, :, :] = img

        with torch.no_grad():
            out = model(x_test)['out']

        out = out.cpu().numpy()[0]
        out = np.argmax(out, axis=0)
        out = to_bgr(out)

        ret = image * 0.6 + out * 0.4
        ret = ret.astype(np.uint8)

        if not os.path.exists('images'):
            os.makedirs('images')

        cv.imwrite('images/{}_image.png'.format(i), image)
        cv.imwrite('images/{}_merged.png'.format(i), ret)
        cv.imwrite('images/{}_out.png'.format(i), out)
