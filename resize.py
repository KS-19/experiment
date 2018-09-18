import os
import glob
import shutil
import cv2
from PIL import Image

def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))

def crop_max_square(pil_img):
    return crop_center(pil_img, min(pil_img.size), min(pil_img.size))

files = sorted(glob.glob('./image_dir/*.png'))

# if not os.path.exists(image_resize):
#     os.makedirs(image_resize)
def save_file_at_new_dir(new_dir_path, new_filename, new_file_content, mode='w'):
    os.makedirs(new_dir_path, exist_ok=True)
    with open(os.path.join(new_dir_path, new_filename), mode) as f:
        f.write(new_file_content)
i=1
for f in files:
    img = Image.open(f)

    im_new = crop_max_square(img)
    #im_new.save('tri000000.png', quality=95)

    img_resize = im_new.resize((64,64))
    # cv2.imwrite('./image_resize'+'img_%s.png' % str(i).zfill(6),img_resize)
    ftitle, fext = os.path.splitext(f)
    img_resize.save('./image_resize/%d.png' %i)# + ftitle + '_28x28' + fext)
    #shutil.move("./image_dir/%s_64x64%s","."%(ftitle,fext))
    i += 1
