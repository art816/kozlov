__author__ = 'art'

from PIL import Image
import numpy as np
import collections as cl
import itertools
import operator


def get_code3(pix_array):
    """

    :param pix_array:
    :return:
    """
    print('get_code')
    code3 = cl.OrderedDict()
    index_of1 = np.where(pix_array == 1)
   # print(pix_array)
    print(len(index_of1[0]))
    for t0 in range(0, len(index_of1[0])):
        for t1 in range(t0+1, len(index_of1[0])):
            for t2 in range(t1+1, len(index_of1[0])):
                x = np.array([index_of1[0][t0], index_of1[0][t1],
                              index_of1[0][t2]])
                y = np.array([index_of1[1][t0], index_of1[1][t1],
                              index_of1[1][t2]])
                pl_area = polyarea(x, y)
                if pl_area:
                    code3[(t0, t1, t2)] = pl_area
    return code3


def get_code4(code3):
    """

    :param code3:
    :return:
    """
    code4 = cl.OrderedDict()
    sorted_code_by_value = sorted(code3.items(),
                                      key=operator.itemgetter(1),
                                      reverse=True)
    keys_list = code3.keys()
    unique_value_in_key = []
    for key in keys_list:
        unique_value_in_key.extend(key)
    unique_value_in_key = sorted(list(set(unique_value_in_key)))
    print(unique_value_in_key)
    for key_for4 in itertools.combinations(unique_value_in_key, 4):
        code4[key_for4] = cl.OrderedDict(code3=cl.OrderedDict())
        for key_for3 in itertools.combinations(key_for4, 3):
            code_value = code3.get(key_for3)
            if code_value is None:
                code4.pop(key_for4)
                break
            else:
                code4[key_for4]['code3'][key_for3] = code_value
    return code4


def polyarea(x, y):
    """

    :param x:
    :param y:
    :return:
    """
    return 0.5*np.abs(np.dot(x, np.roll(y, 1))-np.dot(y, np.roll(x, 1)))


def open_image(image_path):
    """

    :param image_path:
    :return:
    """
    image = Image.open(image_path)
    width = image.size[0]
    height = image.size[1]
    pix_array = np.zeros((height, width))
    image_array = np.asarray(image)
    print(np.max(image_array), np.min(image_array), image_array.shape,
          width, height)
    for num_str in range(height):
        for num_coll in range(width):
            pix_array[num_str, num_coll] = \
                1 if image_array[num_str, num_coll] < 0.5*np.max(image_array) \
                else 0
    return pix_array

if __name__ == '__main__':
    image_path = 'C:\\Users\\art\\Documents\\MATLAB\\Apps\\козлов\\a_s.bmp'
    pix_array = open_image(image_path)
    code3 = get_code3(pix_array)
    print(get_code4(code3))


