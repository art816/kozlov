__author__ = 'art'

from PIL import Image
import numpy as np
import collections as cl
import itertools
import operator

import config as cfg


def get_code3(pix_array):
    """
    :param pix_array:
    :return code3 - OrderedDict с ключами  - tuple (len(tuple) == 3)
        Ключ в code4 состоит из игдексов точек
            трехточечника для которого он получен.
        Значение - площадь трехточечника.
            OrderedDict([((0, 1, 2), 0.5), ((0, 1, 3), 0.5), ((0, 2, 3), 0.5), ((1, 2, 3), 0.5)])
        Пример:

    """
   # print('get_code')
    code3 = cl.OrderedDict()
    index_of1 = np.where(pix_array == 1)
   # print(pix_array)
   # print(len(index_of1[0]))
    for t0 in range(0, len(index_of1[0])):
        for t1 in range(t0+1, len(index_of1[0])):
            for t2 in range(t1+1, len(index_of1[0])):
                x = np.array([index_of1[0][t0], index_of1[0][t1],
                              index_of1[0][t2]])
                y = np.array([index_of1[1][t0], index_of1[1][t1],
                              index_of1[1][t2]])
                pl_area = polyarea(x, y)
                if pl_area >= 2:
                    code3[(t0, t1, t2)] = pl_area
    return code3


def get_code4(code3):
    """
    :param code3:
    :return code4 - OrderedDict с ключами  - tuple (len(tuple) == 4)
        Ключ в code4 состоит из игдексов точек
            четырехточечника для которого он получен.
        Значение - OrderedDict с ключом 'code3'
            Значение - OrderedDict упорядоченный в обратном порядке
                по значению code3 словарь  == code3 где ключи code3
                содержаться в ключе code4.
        Ничего не понятно((.
        Пример:
            OrderedDict([((0, 1, 2, 3), OrderedDict(
                [('code3', OrderedDict(
                    [((0, 1, 2), 1), ((0, 1, 3), 0.9), ((0, 2, 3), 0.7), ((1, 2, 3), 0.5)]
                ))]
            ))])

    """
    code4 = cl.OrderedDict()
    keys_list = code3.keys()
    unique_value_in_key = []
    for key in keys_list:
        unique_value_in_key.extend(key)
    unique_value_in_key = sorted(list(set(unique_value_in_key)))
    #print(unique_value_in_key)
    for key_for4 in itertools.combinations(unique_value_in_key, 4):
        code4[key_for4] = cl.OrderedDict(code3=cl.OrderedDict(),
                                             next_4=None,
                                             investigated=False)
        for key_for3 in itertools.combinations(key_for4, 3):
            code_value = code3.get(key_for3)
            if code_value is None:
                code4.pop(key_for4)
                break
            else:
                code4[key_for4]['code3'][key_for3] = code_value
        #если не было break. sorting by value, reverse
        else:
            code4[key_for4]['code3'] = sorted_sub_dict_code3(code4[key_for4])

    #переупорядочивание словаря по максимальному значению  в словаре code3
    code4 = sorted_dict_code4(code4)
    return code4


def sorted_sub_dict_code3(code):
    """
    :return: Сортируем подсловарь  code3 в порядке
        убывания значения площади.
    """
    return cl.OrderedDict(sorted(code['code3'].items(),
                        key=operator.itemgetter(1), reverse=True))


def sorted_dict_code4(code4):
    """
    :return: сортируем словарь code4 в порядке убывания
        наибольшего элемента в code3
    """
    return cl.OrderedDict(
        sorted(code4.items(),
                # key=lambda t: list(t[1]['code3'].values())[0],
                key=lambda t: sum(t[1]['code3'].values()),
                reverse=True))


def add_key_for_next_small4(code4):
    """
    :param code4:
    :return для каждого четырехточечника добавляет ключ
        следующего четырехугольника, площадь которого меньше, и который
        содержит точки наибольшего трехточечника в составе данного:

        Ничего не понятно((
    """
    keys = list(code4.keys())
    for num_key in range(len(keys)):
        bigger_3 = list(code4[keys[num_key]]['code3'].keys())[0]
        for num_key2 in range(num_key + 1, len(keys)):
            new_4 = list(keys[num_key2])
            new_4.extend(bigger_3)
            if len(set(keys[num_key])) == 4:
                code4[keys[num_key]]['next_4'] = keys[num_key2]
                break


def polyarea(x, y):
    """
    :param x - x-ые координаты:
    :param y - y-ые координаты:
    :return площадь n-угольника заданного векторами x, y:
    """
    return 0.5*np.abs(np.dot(x, np.roll(y, 1))-np.dot(y, np.roll(x, 1)))


def open_image(image_path, rotate=None):
    """
    :param image_path:
    Пока работаем только с изображениями в градациях серого
        (матрица изображения двемерная).
    :return numpy.array соответствующий изображения, где для точек
        с яркость < 0.5*max поставлена в соответствие 1
        с яркость >= 0.5*max поставлен в соответствие 0:
    """
    image = Image.open(image_path)#.convert('L')
    if rotate:
        image = Image.fromarray(image_to_array(image))
        image = image.rotate(cfg.rotate)
        image = Image.fromarray(image_to_array(image))
        #image.show()
    return image_to_array(image)

def image_to_array(image):
    """

    :param image:
    :return:
    """
    width = image.size[0]
    height = image.size[1]
    pix_array = np.zeros((height, width))
    image_array = np.asarray(image)
    # print(image_array.shape, np.max(image_array), image_array)
    #print(np.max(image_array), np.min(image_array), image_array.shape,
    #      width, height)
    for num_str in range(height):
        for num_coll in range(width):
            pix_array[num_str, num_coll] = \
                1 if image_array[num_str, num_coll] < 0.1*np.max(image_array) \
                else 0
    return pix_array


def find_by_code4(target_code4, example_code4):
    """
    
    :param target_code4: 
    :param example_code4: 
    :return:
    """
    target_keys = list(target_code4.keys())
    example_keys = list(example_code4.keys())

    num_finder_example_key = 0
    count_finder_key = 0
    print('len(target_keys)=', len(target_keys), 'len(example_keys)=', len(example_keys))
    for num_target_key in range(len(target_keys)):
        # for num_example_key in range(num_finder_example_key, len(example_keys)):
        # if target_code4[target_keys[num_target_key]]['investigated'] is False:
            for num_example_key in range(len(example_keys)):
                if equal_element_code(
                        list(target_code4[target_keys[num_target_key]]['code3'].values()),
                        list(example_code4[example_keys[num_example_key]]['code3'].values())) and \
                        corr_between_points((target_keys[num_target_key], target_code4[target_keys[num_target_key]]),
                                               (example_keys[num_example_key], example_code4[example_keys[num_example_key]])):
                    current_target_code4 = target_code4[target_keys[num_target_key]]
                    current_example_code4 = example_code4[example_keys[num_example_key]]
                    i = 1
                    list_current_keys_target = [target_keys[num_target_key]]
                    while current_target_code4['next_4'] and current_example_code4['next_4']:
                        i += 1
                        if equal_element_code(
                                list(current_target_code4['code3'].values()),
                                list(current_example_code4['code3'].values())):
                            list_current_keys_target.append(current_target_code4['next_4'])
                            current_target_code4 = target_code4[current_target_code4['next_4']]
                            current_example_code4 = example_code4[current_example_code4['next_4']]
                        else:
                            break
                    else:
                        if not current_example_code4['next_4'] or i >= 2:
                            num_finder_example_key = num_example_key
                            count_finder_key += 1
                            print(num_target_key, num_example_key, i)
                            for current_key in list_current_keys_target:
                                target_code4[current_key]['investigated'] = True
                            break
    # for a,b,c in zip(target_code4.values(), example_code4.values(), range(len(example_code4))):
        # print("{} target {}    example {}".format(c, list(a['code3'].values()), list(b['code3'].values())))
    return len(example_code4.keys()), count_finder_key


def equal_element_code(target_element_code, example_element_code):
    decimals = cfg.decimals
    # print(np.around(
    #         np.array(target_element_code)/target_element_code[0],
    #         decimals=decimals),
    #       np.around(
    #         np.array(example_element_code)/example_element_code[0],
    #         decimals=decimals),
    #       np.array_equal(
    #     np.around(
    #         np.array(target_element_code)/target_element_code[0],
    #         decimals=decimals),
    #     np.around(
    #         np.array(example_element_code)/example_element_code[0],
    #         decimals=decimals)))
    return np.array_equal(
        np.around(
            np.array(target_element_code)/target_element_code[0],
            decimals=decimals),
        np.around(
            np.array(example_element_code)/example_element_code[0],
            decimals=decimals))


def find_transform(target, example):
    A = np.linalg.lstsq(example, target)[0]
    #A = np.linalg.solve(example, target)
    return A, np.std(target - np.dot(example, A))


def corr_between_points(target, example):
    """

    :param target:
    :param example:
    :return:
    """
    value_count_target = cl.OrderedDict()
    value_count_example = cl.OrderedDict()
    # print(target[0], target[1])
    for value_in_key in target[0]:
        value_count_target[value_in_key] = [value_in_key in value_in_key_code3
            for value_in_key_code3 in target[1]['code3'].keys()]
    value_count_target = cl.OrderedDict(
        sorted(value_count_target.items(),
                key=operator.itemgetter(1)))
    for value_in_key in example[0]:
        value_count_example[value_in_key] = [value_in_key in value_in_key_code3
            for value_in_key_code3 in example[1]['code3'].keys()]
    value_count_example = cl.OrderedDict(
        sorted(value_count_example.items(),
                key=operator.itemgetter(1)))
    # если нашлось соответствие возвращаем список из пар точек,
    # иначе None
    if ([value for value in value_count_target.values()] ==
            [value for value in value_count_example.values()]):
        return list(zip(value_count_target.keys(), value_count_example.keys()))





if __name__ == '__main__':
    image_path = 'C:\\Users\\art\\Documents\\MATLAB\\Apps\\козлов\\a_s.bmp'
    pix_array = open_image(image_path)
    code3 = get_code3(pix_array)
    #print(get_code4(code3))


