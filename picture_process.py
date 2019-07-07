from PIL import Image
import os

origin_path = 'E:\\Dataset\\上传\\苹果\\苹果4-157除313=0.50159\\'
dst_path = 'E:\\Dataset\\上传\\Apple\\'
for dir_name in os.listdir(origin_path):
    full_path = os.path.join(origin_path, dir_name)
    count = 1
    for pic in os.listdir(full_path):
        im = Image.open(os.path.join(full_path, pic))
        out = im.resize((256, 256), Image.ANTIALIAS)
        out.save(dst_path + dir_name + '-' + str(count) + '.jpg')
        count = count + 1
    print('finish ' + dir_name)

# infile = 'D:\\original_img.jpg'
# outfile = 'D:\\adjust_img.jpg'
# im = Image.open(infile)
# (x, y) = im.size  # read image size
# x_s = 250  # define standard width
# y_s = y * x_s / x  # calc height based on standard width
# out = im.resize((x_s, y_s), Image.ANTIALIAS)  # resize image with high-quality
# out.save(outfile)
