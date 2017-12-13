# input: image include some digits
# output: digits label
import math
import os
import random
import sys

import cv2
import numpy
import numpy as np

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

OUTPUT_SHAPE = (32, 120)
DIGITS = '0123456789'

LENGTHS = [3,4,5]
max_n_len = 7
FONT_HEIGHT = 28
fonts = ['fonts/huawenxihei.ttf']
bg_path = 'bgs/'

bg_file_list = os.listdir(bg_path)
bg_nums = len(bg_file_list)

def sparse_tuple_from(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n]*len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)

    return indices, values, shape

def euler_to_mat(yaw, pitch, roll):
    # Rotate clockwise about the Y-axis
    c, s = math.cos(yaw), math.sin(yaw)
    M = numpy.matrix([[c, 0., s],
                      [0., 1., 0.],
                      [-s, 0., c]])

    # Rotate clockwise about the X-axis
    c, s = math.cos(pitch), math.sin(pitch)
    M = numpy.matrix([[1., 0., 0.],
                      [0., c, -s],
                      [0., s, c]]) * M

    # Rotate clockwise about the Z-axis
    c, s = math.cos(roll), math.sin(roll)
    M = numpy.matrix([[c, -s, 0.],
                      [s, c, 0.],
                      [0., 0., 1.]]) * M

    return M


def make_affine_transform(from_shape, to_shape,
                          min_scale, max_scale,
                          scale_variation=1.0,
                          rotation_variation=1.0,
                          translation_variation=1.0):
    out_of_bounds = False

    from_size = numpy.array([[from_shape[1], from_shape[0]]]).T
    to_size = numpy.array([[to_shape[1], to_shape[0]]]).T

    scale = random.uniform((min_scale + max_scale) * 0.5 -
                           (max_scale - min_scale) * 0.5 * scale_variation,
                           (min_scale + max_scale) * 0.5 +
                           (max_scale - min_scale) * 0.5 * scale_variation)

    if scale > max_scale or scale < min_scale:
        out_of_bounds = True
    roll = random.uniform(-0.3, 0.3) * rotation_variation
    pitch = random.uniform(-0.2, 0.2) * rotation_variation
    yaw = random.uniform(-1.2, 1.2) * rotation_variation
    # Compute a bounding box on the skewed input image (`from_shape`).
    M = euler_to_mat(yaw, pitch, roll)[:2, :2]
    h, w = from_shape[0:2]
    corners = numpy.matrix([[-w, +w, -w, +w],
                            [-h, -h, +h, +h]]) * 0.5
    skewed_size = numpy.array(numpy.max(M * corners, axis=1) -
                              numpy.min(M * corners, axis=1))

    # Set the scale as large as possible such that the skewed and scaled shape
    # is less than or equal to the desired ratio in either dimension.
    scale *= numpy.min(to_size / skewed_size) * 1.1

    # Set the translation such that the skewed and scaled image falls within
    # the output shape's bounds.
    trans = (numpy.random.random((2, 1)) - 0.5) * translation_variation
    trans = ((2.0 * trans) ** 5.0) / 2.0
    if numpy.any(trans < -0.5) or numpy.any(trans > 0.5):
        out_of_bounds = True
    trans = (to_size - skewed_size * scale) * trans

    center_to = to_size / 2.
    center_from = from_size / 2.

    M = euler_to_mat(yaw, pitch, roll)[:2, :2]

    M *= scale
    M = numpy.hstack([M, trans + center_to - M * center_from])

    return M, out_of_bounds


# -----------------generate label----------------
def generate_label(length):
    f = ""
    #length = random.choice(LENGTHS)

    for _ in range(length):
        f = f + random.choice(DIGITS)
    return f


# ----------------generate bg---------------------
def generate_bg(bg_pic_num=bg_nums):
    while True:
        fname = "bgs/{:08d}.png".format(random.randint(0, bg_pic_num - 1))
        bg = cv2.imread(fname) / 255.0

        if (bg.shape[1] >= OUTPUT_SHAPE[1] and bg.shape[0] >= OUTPUT_SHAPE[0]):
            x = random.randint(0, bg.shape[1] - OUTPUT_SHAPE[1])
            y = random.randint(0, bg.shape[0] - OUTPUT_SHAPE[0])
            bg = bg[y:y + OUTPUT_SHAPE[0], x:x + OUTPUT_SHAPE[1]]
            break

    return bg


# ---------------------make char image----------------------------
def make_char_ims(output_height, font):
    font_size = output_height * 1
    font = ImageFont.truetype(font, font_size)
    height = max(font.getsize(d)[1] for d in DIGITS)
    for c in DIGITS:
        width = font.getsize(c)[0]
        im = Image.new("RGBA", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(im)
        draw.text((0, 0), c, (0, 0, 0), font=font)
        scale = float(output_height) / height
        im = im.resize((int(width * scale), int(output_height * scale)), Image.ANTIALIAS)
        yield c, numpy.array(im)[:, :, 0].astype(numpy.float32) / 255.


def get_all_font_char_ims(out_height):
    result = []
    for font in fonts:
        result.append(dict(make_char_ims(out_height, font)))
    return result


def generate_plate(font_height, char_ims, text_color, length):
    h_padding = 0  # random.uniform(0.2, 0.3) * font_height#(0.2,0.4)
    v_padding = h_padding  # random.uniform(0.1, 0.3) * font_height
    spacing = font_height * random.uniform(0.01, 0.05)
    radius = 1  # + int(font_height * 0.1 * random.random())
    code = generate_label(length)
    text_width = sum(char_ims[c].shape[1] for c in code)
    text_width += (len(code) - 1) * spacing

    out_shape = (int(font_height + v_padding * 2),
                 int(text_width + h_padding * 2), 3)

    text_mask = numpy.ones(out_shape)

    x = h_padding
    y = v_padding
    pos = np.zeros(shape=(len(code), 2), dtype=np.float)
    for index, c in enumerate(code):
        char_im = char_ims[c]
        # print(char_im.shape)
        ix, iy = int(x), int(y)
        text_mask[iy:iy + char_im.shape[0], ix:ix + char_im.shape[1], 0] = char_im
        text_mask[iy:iy + char_im.shape[0], ix:ix + char_im.shape[1], 1] = char_im
        text_mask[iy:iy + char_im.shape[0], ix:ix + char_im.shape[1], 2] = char_im
        x += char_im.shape[1] + spacing
        pos[index][0] = ix
        pos[index][1] = iy
    plate = (text_color * text_mask)

    return plate, code, pos


def generate_im(char_ims, length):
    text_color = random.uniform(0.5, 1.0)
    bg = generate_bg()
    while True:
        plate, label, pos = generate_plate(FONT_HEIGHT, char_ims, text_color, length)
        M, out_of_bounds = make_affine_transform(
            from_shape=plate.shape,
            to_shape=bg.shape,
            min_scale=0.9,
            max_scale=1.0,
            rotation_variation=0.02,
            scale_variation=1.0,
            translation_variation=0.4)

        mask = np.ones((plate.shape[0], plate.shape[1], 3), dtype=np.float32)
        plate = cv2.warpAffine(plate, M, (bg.shape[1], bg.shape[0]))
        mask = cv2.warpAffine(mask, M, (bg.shape[1], bg.shape[0]))

        out = plate * bg + text_color * bg * (1 - mask)
        out = cv2.resize(out, (OUTPUT_SHAPE[1], OUTPUT_SHAPE[0]))
        out = numpy.clip(out, 0., 1.)

        if M[0, 2] > 0:
            break
    return out, label


def name_training_data_generator(batch_size=32):
    a = get_all_font_char_ims(31)
    # print(type(a[0]))
    XX = np.zeros((batch_size,  OUTPUT_SHAPE[0], OUTPUT_SHAPE[1], 1), dtype=np.float32)
    # print(OUTPUT_SHAPE[0])
    # YY = np.zeros((batch_size, OUTPUT_SHAPE[1]), dtype=np.float32)

    label_length = np.ones(batch_size)
    # print(Y)
    label_len = np.zeros(batch_size, dtype=np.int64)
    while True:
        length = random.choice(LENGTHS)
        n_len = length
        Y = []
        #Y = np.ones((batch_size, max_n_len), dtype=np.float32) * -2
        YY = np.ones((batch_size, n_len - 1), dtype=np.float32) * -2
        for i in range(batch_size):
            img, label = generate_im(a[0], length)
            label_length[i] = len(label)

            blur_rand = random.randint(0, 4)
            kernel_size = random.randint(0, 2) * 2 + 1

            if blur_rand != 0:
                img = img * 255.0
                img = img.astype(np.uint8)
                img = cv2.medianBlur(img, kernel_size)
                img = img / 255.0
            blur_rand = random.randint(0, 4)
            kernel_size = random.randint(0, 2) * 2 + 1
            sigma = random.randint(0, 3)
            if blur_rand != 0:
                img = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)

            img_gray = 0.11*img[:,:,0] + 0.59*img[:,:,1] + 0.3*img[:,:,2]
            img_gray = img_gray[..., np.newaxis]
            XX[i] = img_gray
            codes = []
            for index, code in enumerate(label):
                codes.append(DIGITS.find(code))
            Y.append(codes)

        sparse_Y = sparse_tuple_from(Y)
        #print(sparse_Y)
        yield {'input': XX, 'label': sparse_Y, 'feature_length': np.ones(batch_size)*29}


if __name__ == '__main__':
    batch_size = 2
    r = next(name_training_data_generator(batch_size))
    for i in range(batch_size):
        cv2.imwrite(str(i)+'.jpg', 255*r['input'][i])
        print(r['label'][i])


