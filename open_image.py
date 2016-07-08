__author__ = 'art'

from PIL import Image
import numpy as np
import collections as cl
import itertools
import operator


def get_code(pix_array):
    """"""
    print('get_code')
    code = cl.OrderedDict()
    index_of1 = np.where(pix_array == 1)
   # print(pix_array)
    print(len(index_of1[0]))
    for t0 in range(0, len(index_of1[0])):
        for t1 in range(t0+1, len(index_of1[0])):
            for t2 in range(t1+1, len(index_of1[0])):
                x = np.array([index_of1[0][t0], index_of1[0][t1], index_of1[0][t2]])
                y = np.array([index_of1[1][t0], index_of1[1][t1], index_of1[1][t2]])
                pl_area = polyarea(x, y)
                if pl_area:
                    code[(t0, t1, t2)] = pl_area
    return code

def get_4code(code):
    """"""
    code4 = cl.OrderedDict()
    sorted_code_by_value = sorted(code.items(), key=operator.itemgetter(1), reverse=True)
    keys_list = code.keys()
    unique_value_in_key = []
    for key in keys_list:
        unique_value_in_key.extend(key)
    unique_value_in_key = sorted(list(set(unique_value_in_key)))
    print(unique_value_in_key)
    for key_for4 in itertools.combinations(unique_value_in_key, 4):
        code4[key_for4] = cl.OrderedDict(code=cl.OrderedDict())
        for key_for3 in itertools.combinations(key_for4, 3):
            code_value = code.get(key_for3)
            if code_value is None:
                code4.pop(key_for4)
                break
            else:
                code4[key_for4]['code'][key_for3] = code_value
    print(code4)



def polyarea(x, y):
    return 0.5*np.abs(np.dot(x, np.roll(y, 1))-np.dot(y, np.roll(x, 1)))


if __name__ == '__main__':
    print('main')
    image = Image.open('C:\\Users\\art\\Documents\\MATLAB\\Apps\\козлов\\a_s.bmp')
    width = image.size[0]
    height = image.size[1]
    pix = image.load()

    pix_array = np.zeros((height, width))
    I = np.asarray(image)
    print(np.max(I), np.min(I), I.shape, width, height)


    for num_str in range(height):
        for num_coll in range(width):
            pix_array[num_str, num_coll] = 1 if I[num_str, num_coll] < 8 else 0

    #index_of1 = np.where(pix_array == 1)
    code = get_code(pix_array)
    get_4code(code)


