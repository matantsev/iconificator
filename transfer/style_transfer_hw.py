from __future__ import print_function
import torch

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

from transfer.model.global_props import device
from transfer.run_transfer import run_style_transfer

# desired size of the output image
imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu
loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor


def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    # image = loader(image).unsqueeze(0)
    # return image.to(device, torch.float)
    return image

def crop_to_image(img_src,img_crop):
    src_width, src_height = img_src.size # Get dimensions

    return img_crop.resize((src_width, src_height), Image.ANTIALIAS)


def format_image(img):
    return loader(img).unsqueeze(0).to(device, torch.float)


style_img = image_loader("./icon_samples/2.jpg")

# style_img = image_loader("./transfer/images/picasso.jpg")

# content_img = image_loader("./transfer/images/do_not_push.jpg")

content_img = image_loader("./transfer/images/picasso.jpg")
# assert style_img.size() == content_img.size(), \
#     "we need to import style and content images of the same size"
if style_img.size != content_img.size:
    style_img = crop_to_image(content_img, style_img)
style_img = format_image(style_img)
content_img = format_image(content_img)

unloader = transforms.ToPILImage()  # reconvert into PIL image

plt.ion()


def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated


plt.figure()
imshow(style_img, title='Style Image')

plt.figure()
imshow(content_img, title='Content Image')

cnn = models.vgg19(pretrained=True).features.to(device).eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

input_img = content_img.clone()
plt.figure()

output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, input_img)

plt.figure()
imshow(output, title='Output Image')

# sphinx_gallery_thumbnail_number = 4
plt.ioff()
plt.show()