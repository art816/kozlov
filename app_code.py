__author__ = 'art'

import collections as cl
import itertools
import operator

import cv2
import numpy as np
from PIL import Image

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
    #TODO вынести отдельно поиск ненулевых элементов
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
    return code3, index_of1


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
                                         investigated=False,
                                         parent=None,
                                         level=0
                                         )
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
    unique_value_in_key = []
    for key in keys:
        unique_value_in_key.extend(key)
    unique_value_in_key = sorted(list(set(unique_value_in_key)))
    for num_key in range(len(keys)):
        bigger_3 = list(code4[keys[num_key]]['code3'].keys())[0]
        for num_key2 in range(num_key + 1, len(keys)):
            new_4 = list(keys[num_key2])
            new_4.extend(bigger_3)
            # new_3 = list(code4[keys[num_key2]]['code3'].keys())[0]
            # new_3.extend(bigger_3)
            # if len(set(keys[num_key])) == 4:
            # if len(set(keys[num_key])) == 4:
            # Добавляем ссылки на родителей и детей.
            # Один ребенок, несколько родителей.
            if len(set(new_4)) == 4:
        # for new_value in unique_value_in_key:
        #     if new_value not in bigger_3:
        #         next_4 = list(bigger_3)
        #         next_4.append(new_value)
        #         next_4 = sorted(next_4)
                
                code4[keys[num_key]]['next_4'] = keys[num_key2]
                if code4[keys[num_key2]]['parent'] is None:
                    code4[keys[num_key2]]['parent'] = [keys[num_key]]
                else:
                    code4[keys[num_key2]]['parent'].append(keys[num_key])
                break

    # Проходим от самых маленьких к большим n-угольникам и выставляем уровни(число потомков)
    for num_key in reversed(range(len(keys))):
        parent_list = code4[keys[num_key]]['parent']
        if parent_list:
            for parent_key in parent_list:
                code4[parent_key]['level'] = code4[keys[num_key]][
                                             'level'] + 1


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
    image = Image.open(image_path).convert('L')
    if rotate:
        # Преобразование image_to_array чтобы инвертировать цвета,
        # при повороте добавляется черный фон.
        image = Image.fromarray(image_to_array(image))
        image = image.rotate(cfg.rotate, expand=1)
        # new_size = cfg.resize[:2] * np.array([image.size[0], image.size[1]])
        # new_size = new_size.astype(int)
        # print(new_size)
        # image = image.resize(new_size, resample=0)
        image = Image.fromarray(image_to_array(image))
        # image.show()
    return image_to_array(image)

def image_to_array(image):
    """

    :param image:
    :return:
    """
    width = image.size[0]
    height = image.size[1]
    pix_array = np.zeros((height, width), 'uint8')
    image_array = np.asarray(image)
    for num_str in range(height):
        for num_coll in range(width):
            #TODO выбирать только особые точки, а то слишком много
            pix_array[num_str, num_coll] = \
                1 if image_array[num_str, num_coll] < cfg.img_level_binary*np.max(image_array) \
                else 0
    return pix_array


def find_futures(pix_array, num_futures):
    """

    :param pix_array:
    :return:
    """
    dst = cv2.cornerHarris(pix_array, 5, 3, 0.01)
    new_array = np.zeros((pix_array.shape[0], pix_array.shape[1], 3), 'uint8')
    sort_dst = np.sort(abs(dst), axis=None)
    for i in range(pix_array.shape[0]):
        for j in range(pix_array.shape[1]):
            new_array[i][j] = [0, 0, 255] if abs(dst[i][j])>sort_dst[-num_futures] \
                else np.array([255, 0, 0])*pix_array[i][j]
    return new_array


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
    list_pairs = []
    print('len(target_keys)=', len(target_keys), 'len(example_keys)=', len(example_keys))
    for num_target_key in range(len(target_keys)):
        # for num_example_key in range(num_finder_example_key, len(example_keys)):
        # if target_code4[target_keys[num_target_key]]['investigated'] is False:
        #     if list(example_code4[example_keys[num_example_key]]['code3'].values)[0]/list(target_code4[target_keys[num_target_key]]['code3'].values)[0] > cfg.diff_size:
                # break
            if target_code4[target_keys[num_target_key]]['level'] < cfg.num_equal:
                continue

            for num_example_key in range(len(example_keys)):
                num_iteration = 0
                num_equals = 0
                if example_code4[example_keys[num_example_key]]['level'] < cfg.num_equal:
                    continue
                curr_target_key = target_keys[num_target_key]
                curr_example_key = example_keys[num_example_key]
                current_target_code4 = target_code4[curr_target_key]
                current_example_code4 = example_code4[curr_example_key]
                list_current_keys_target = []
                pair_equal_code = []
                curr_pair_equal_code = equal_code(
                        (curr_target_key, current_target_code4),
                        (curr_example_key, current_example_code4))
                if curr_pair_equal_code:
                    pair_equal_code.extend(curr_pair_equal_code)
                    list_current_keys_target.append(curr_target_key)
                    curr_target_key = current_target_code4['next_4']
                    curr_example_key = current_example_code4['next_4']

                    num_iteration = 1
                    num_equals = 1
                    while curr_target_key and curr_example_key:
                        num_iteration += 1
                        if target_code4[curr_target_key]['level'] <\
                                cfg.num_equal - num_equals\
                            or\
                            example_code4[curr_example_key]['level'] <\
                                cfg.num_equal - num_equals:
                            break
                        current_target_code4 = target_code4[curr_target_key]
                        current_example_code4 = example_code4[curr_example_key]
                        curr_pair_equal_code = equal_code(
                            (curr_target_key, current_target_code4),
                            (curr_example_key, current_example_code4))
                        if curr_pair_equal_code:
                            pair_equal_code.extend(curr_pair_equal_code)
                            list_current_keys_target.append(curr_target_key)
                            curr_target_key = current_target_code4['next_4']
                            curr_example_key = current_example_code4['next_4']
                            num_equals += 1
                        else:
                            curr_target_key = current_target_code4['next_4']

                    count_finder_key += 1
                    for current_key in list_current_keys_target:
                        target_code4[current_key]['investigated'] = True
                    if num_equals >= cfg.num_equal:
                        list_pairs.append(pair_equal_code)
                        # break

                        # return [pair_equal_code]
    return list_pairs


def equal_code(target, example):
    """

    :param target: tuple(key, value)
    :param example: tuple(key, value)
    :return: список из пар соответствующих точек если коды равны и
               нашлось соответствие между точками. Иначе None
    """
    if equal_element_code(
            list(target[1]['code3'].values()),
            list(example[1]['code3'].values())):
        return corr_between_points(target, example)

def equal_element_code(target_element_code, example_element_code):
    """

    :param target_element_code:
    :param example_element_code:
    :return: True если коды равны с указаной точностью.
               Иначе False. или отношение площадей
    """
    decimals = cfg.decimals
    if max(
            target_element_code[0]/example_element_code[0],
            example_element_code[0]/target_element_code[0]) < cfg.diff_size:
        return np.array_equal(
            np.around(
                np.array(target_element_code)/target_element_code[0],
                decimals=decimals),
            np.around(
                np.array(example_element_code)/example_element_code[0],
                decimals=decimals))
    # else:
    #     return target_element_code[0]/example_element_code[0]


def find_transform(target, example):
    A = np.linalg.lstsq(example, target)[0]
    #A = np.linalg.solve(example, target)
    return A, [np.sum(abs(target - np.dot(example, A)))/target.shape[0], target.shape[0]]


def corr_between_points(target, example):
    """

    :param target:
    :param example:
    :return:  список из пар соответствующих точек если
               нашлось соответствие между точками. Иначе None.
    """
    # TODO Возможна неправильная работа если все площади равны
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
    # print(value_count_target, '\n', value_count_example)
    if ([value for value in value_count_target.values()] ==
            [value for value in value_count_example.values()]):
        return list(zip(value_count_target.keys(), value_count_example.keys()))

def transform(index_of1, size_array):
    """
    Transform without interpolation.
    :param index_of1:
    :return:
    """
    import math

    angle = -cfg.rotate * np.pi / 180
    A = np.array([[np.cos(angle), -np.sin(angle), 0],
                  [np.sin(angle), np.cos(angle), 0],
                  [0, 0, 1]])
    # A = np.array([[1, 0, 0],
    #              [0, 1, 0],
    #              [0, 0, 1]])

    A = A * cfg.resize
    # calculate output size
    w, h = np.array(size_array)
    xx = []
    yy = []
    for x, y in ((0, 0), (w, 0), (w, h), (0, h)):
        example = np.vstack([x, y, 1]).T
        xy = np.dot(example, A)
        xx.append(xy[0][0])
        yy.append(xy[0][1])
    w = int(math.ceil(max(xx)) - math.floor(min(xx)))
    h = int(math.ceil(max(yy)) - math.floor(min(yy)))
    example = np.vstack([size_array[0] / 2.0, size_array[1] / 2.0, 1]).T
    xy = np.dot(example, A)
    A[2, 0] = w / 2 - xy[0][0]
    A[2, 1] = h / 2 - xy[0][1]

    # A = A.T * cfg.resize
    #TODO size, пересчет размеров с учетом поворота!!!!!!!!!!!???
    new_size = np.array([w, h])#np.array(size_array) * cfg.resize[:2] * 2
    new_size = new_size.astype(int)
    pix_array = np.zeros(new_size)
    for t0 in range(0, len(index_of1[0])):
        x = np.array(index_of1[0][t0])
        y = np.array(index_of1[1][t0])
        example = np.vstack([x, y, 1]).T
        xy = np.dot(example, A)
        # pix_array[
        #     int(xy[0][0]),
        #     int(xy[0][1])] = 1
        pix_array[
            int(xy[0][0])%new_size[0],
            int(xy[0][1])%new_size[1]] = 1
        # pix_array[int(xy[0][0]) if xy[0][0] < new_size[0] - 1 else new_size[0] - 1,
        #            int(xy[0][1]) if xy[0][1] < new_size[1] - 1 else new_size[1] - 1] = 1
    return pix_array

def select_point(pix_array, random=0):
    """

    :param pix_array:
    :return:
    """
    # print(pix_array.shape, pix_array.sum())
    top = int(pix_array.sum()/cfg.selected_points) + 1
    top = top if top >= 1 else 1
    pix_array
    indices = list(pix_array.nonzero())
    # print(top, indices[::top])
    selected_pix_array = np.zeros(pix_array.shape)

    # print([indices[0][::top], indices[1][::top]])
    # Перемешиваем, чтобы равномернее выбирать точки изображения
    # shuffled_indeces = np.random.permutation(np.arange(indices[0].size))
    # indices[0] = indices[0][shuffled_indeces]
    # indices[1] = indices[1][shuffled_indeces]
    # np.random.shuffle()
    # random_choice = np.random.choice(
    #     indices[0].size, cfg.selected_points, replace=False)
    # selected_pix_array[indices[0][random_choice], indices[1][random_choice]] = 1
    if random:
        selected_pix_array[
            indices[0][::top] + np.random.random_integers(-int(random/2), int(random/2), cfg.selected_points),
            indices[1][::top] + np.random.random_integers(-int(random/2), int(random/2), cfg.selected_points)] = 1
    else:
        selected_pix_array[indices[0][::top], indices[1][::top]] = 1
    # print(selected_pix_array.sum())
    # print(selected_pix_array)
    return selected_pix_array

def get_dict_pairs(pairs):
    """

    :param pairs: list of pairs eqaul ponts
    :return: dict pairs sorted by truth
    """
    dict_pairs = dict()
    for target_n, example_n in pairs:
        if dict_pairs.get((target_n, example_n)):
            dict_pairs[(target_n, example_n)] = dict_pairs[
                                                    (target_n, example_n)] + 1
        else:
            dict_pairs[(target_n, example_n)] = 1
    dict_pairs = cl.OrderedDict(
        sorted(dict_pairs.items(), key=operator.itemgetter(1), reverse=True))
    return dict_pairs

def get_unique_pairs_by_truth(dict_pairs):
    """

    :param dict_pairs:
    :return:
    """
    unique_target = []
    unique_example = []
    unique_pairs = []
    for current_targen_num, current_example_num in dict_pairs.keys():
        if current_targen_num in unique_target or current_example_num in unique_example:
            pass
        else:
            unique_pairs.append((current_targen_num, current_example_num))
            unique_target.append(current_targen_num)
            unique_example.append(current_example_num)
    return unique_pairs


def get_img_chanel(img, chanel):
    """
    :param img:
    :param chanel:
    :return:
    """
    pix_array = np.zeros(img.shape[:2])
    for num_str in range(img.shape[0]):
        for num_coll in range(img.shape[1]):
            # TODO выбирать только особые точки, а то слишком много
            red = img[num_str, num_coll, 0]
            green = img[num_str, num_coll, 1]
            blue = img[num_str, num_coll, 2]
            if chanel == 'red':
                pix_array[
                    num_str, num_coll] = 1 if\
                        red >= 126 and green < 126 and blue < 126\
                    else 0
            elif chanel == 'blue':
                pix_array[
                    num_str, num_coll] = 1 if\
                        red < 126 and green < 126 and blue >= 126\
                    else 0
            elif chanel == 'green':
                pix_array[
                    num_str, num_coll] = 1 if\
                        red < 126 and green >= 126 and blue < 126\
                    else 0
    return pix_array

def indecs_1d_to_2d(indices_1d, index_of1):
    """

    :param unique_pairs:
    :param target_index_of1:
    :param example_index_of1:
    :return:
    """
    coordinat_x = []
    coordinat_y = []
    for coordinat_n in indices_1d:
        coordinat_x.append(index_of1[0][coordinat_n])
        coordinat_y.append(index_of1[1][coordinat_n])
    indices_2d = np.vstack(
        [coordinat_x, coordinat_y, np.ones(len(coordinat_x))]).T
    return indices_2d


if __name__ == '__main__':
    image_path = 'C:\\Users\\art\\Documents\\MATLAB\\Apps\\козлов\\a_s.bmp'
    pix_array = open_image(image_path)
    code3 = get_code3(pix_array)
    #print(get_code4(code3))


