import os
from torchvision import transforms

def save_img(x, suffix):
    img = x.cpu().clone()
    img = img.squeeze(0)
    img = transforms.ToPILImage()(img)
    img_save_dir = './result'
    if not os.path.exists(img_save_dir):
        os.mkdir(img_save_dir)
    suffix = suffix.replace('jpg', 'png')
    img.save(os.path.join(img_save_dir, suffix))

