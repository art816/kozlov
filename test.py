__author__ = 'art'

import unittest
import numpy as np
import itertools
import copy
from PIL import Image
import matplotlib.pyplot as plt
import os
from multiprocessing import Process


import app_code
import config as cfg


class Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        :return:
        """
        cls.x = np.array([0, 0, 1, 1])
        cls.y = np.array([0, 1, 1, 0])
        cls.image_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), cfg.image_names[0])
        cls.pix_array = app_code.open_image(cls.image_path)
        cls.code3, cls.index_of1 = app_code.get_code3(
            cls.pix_array)
        cls.code4 = app_code.get_code4(cls.code3)

    def tearDown(self):
        """
        :return:
        """
        pass

    def test_polyarea(self):
        """
        :return:
        """
        self.assertEqual(app_code.polyarea(self.x, self.y), 1)

    def test_open_image(self):
        """
        :return:
        """
        self.assertIsNotNone(app_code.open_image(self.image_path))


    def test_get_code3(self):
        """
        :return:
        """
        code3, index_of1 = app_code.get_code3(
            np.array([self.x, self.y]))
        self.assertEqual(len(code3.keys()), 4)
        for key in itertools.combinations(list(range(4)), 3):
            self.assertEqual(code3.get(key), 0.5)

    def test_get_code4(self):
        """
        :return:
        """
        first_key = list(self.code4.keys())[0]
        # self.assertEqual(self.self.code4.get(first_key)['code3'], self.code3)
        code4_values = list(self.code4.get(first_key)['code3'].values())

        self.assertEqual(
            code4_values, sorted(code4_values, reverse=True))
        code4_first_values_for_key = [list(self.code4[key]['code3'].values())[0] for key in self.code4]
        self.assertTrue(
            code4_first_values_for_key == sorted(code4_first_values_for_key, reverse=True))

    def test_find_by_code4(self):
        """
        :return:
        """
        #TODO сделать нормальный тест, разбить на части
        code3 = app_code.get_code3(
             np.array([self.x, self.y]))
        example_code4 = app_code.get_code4(self.code3)

        # target_pix_array = app_code.open_image(self.image_path, rotate=0)#'C:\\Users\\art\\Documents\\MATLAB\\Apps\\козлов\\ABVG.bmp', rotate=0)
        target_pix_array = app_code.open_image(os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg.image_names[0]), rotate=0)

        target_code3, target_index_of1 = app_code.get_code3(
            target_pix_array)
        target_pix_array = app_code.transform(target_index_of1,
                                                   target_pix_array.shape)
        target_code3, target_index_of1 = app_code.get_code3(
            target_pix_array)
        target_code4 = app_code.get_code4(target_code3)
        app_code.add_key_for_next_small4(target_code4)
        app_code.add_key_for_next_small4(example_code4)
        #TODO открывать изображения в новом потоке
        # Process(target=plt.show, args=(None,)).start()
        # target_code4 = copy.deepcopy(example_code4)
        plt.figure(1)
        plt.imshow(target_pix_array, interpolation='none')
        plt.figure(2)
        plt.imshow(self.pix_array, interpolation='none')
        plt.show()
        pairs_equal_code = app_code.find_by_code4(target_code4, example_code4)
        if pairs_equal_code:
            # TODO помним что есть повторяющиеся пары в коде.
            #  Преобразование строится методом наименьших квадратов.
            # К повторяющимся парам больше доверия
            print(len(pairs_equal_code), pairs_equal_code)
            for pairs in pairs_equal_code:
                target_x = []
                target_y = []
                example_x = []
                example_y = []
                # pairs = pairs_equal_code[0]
                for target_n, example_n in pairs:
                    target_x.append(target_index_of1[0][target_n])
                    target_y.append(target_index_of1[1][target_n])
                    example_x.append(self.index_of1[0][example_n])
                    example_y.append(self.index_of1[1][example_n])
                target = np.vstack([target_x, target_y, np.ones(len(target_x))]).T
                example = np.vstack([example_x, example_y, np.ones(len(example_x))]).T
                A, std = app_code.find_transform(target, example)
                print(A, std, type(std))
                if std > cfg.std:
                    continue
                pix_array = np.zeros(target_pix_array.shape)
                # print(image_array.shape, np.max(image_array), image_array)
                #print(np.max(image_array), np.min(image_array), image_array.shape,
                #      width, height)
                pix_array_base = np.zeros(target_pix_array.shape, 'uint8')
                xy_base = np.dot(example, A)
                for n in range(xy_base.shape[0]):
                    pix_array_base[int(round(xy_base[n][0])), int(round(xy_base[n][1]))] = 1

                for t0 in range(0, len(self.index_of1[0])):
                    x = np.array(self.index_of1[0][t0])
                    y = np.array(self.index_of1[1][t0])
                    example_one = np.vstack([x, y, 1]).T
                    xy = np.dot(example_one, A)
                    pix_array[int(round(xy[0][0])), int(round(xy[0][1]))] = 1
                rgbArray = np.ones((target_pix_array.shape[0], target_pix_array.shape[1], 3), 'uint8')
                print(rgbArray.shape)
                rgbArray[..., 0] = 255*(target_pix_array)
                rgbArray[..., 1] = 255*(pix_array) - 127*pix_array_base
                rgbArray[..., 2] = 127*np.ones(target_pix_array.shape, 'uint8')
                plt.figure(3)
                plt.imshow(rgbArray, interpolation='none')
                #TODO открывать изображения в новом потоке
                plt.show()

    def test_find_futures(self):
        """

        :return:
        """
        pix_array = app_code.find_futures(self.pix_array, num_futures=5)
        plt.imshow(pix_array, interpolation='none')
        plt.show()


    def test_transform(self):
        """

        :return:
        """
        target_pix_array = app_code.open_image(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), cfg.image_names[1]), rotate=0)
        plt.figure(1)
        plt.imshow(target_pix_array, interpolation='none')
        target_code3, target_index_of1 = app_code.get_code3(
            target_pix_array)
        transform_pix_array = app_code.transform(target_index_of1, target_pix_array.shape)
        plt.figure(2)
        plt.imshow(transform_pix_array, interpolation='none')
        plt.show()



    def test_find_transform(self):
        """
        :return:
        """
        target = np.vstack([self.x, self.y, np.ones(len(self.x))]).T
        example = np.vstack([self.x, self.y, np.ones(len(self.x))]).T
        A, std = app_code.find_transform(target, example)
        self.assertIsNotNone(A)

    def test_add_key_for_next_small4(self):
        """

        :return:
        """
        code4 = copy.deepcopy(self.code4)
        app_code.add_key_for_next_small4(code4)
        for key in code4:
            if code4[key]['next_4']:
                self.assertGreaterEqual(
                    list(code4[key]['code3'].values())[0],
                    list(code4[code4[key]['next_4']]['code3'].values())[0])
                # list(code4[key]['code3'].values())
                # list(code4[code4[key]['next_4']]['code3'].values())
                # self.assertEqual(len(set(list(list(code4[key]['code3'].keys())[0]).extend(list(code4[code4[key]['next_4']]['code3'].keys())[0]))))

    def test_corr_between_points(self):
        """

        :return:
        """
        self.assertIsNotNone(app_code.corr_between_points(
            list(self.code4.items())[0], list(self.code4.items())[0]))
    
    def test_equal_element_code(self):
        """

        :return:
        """
        self.assertTrue(app_code.equal_element_code(
            list(list(self.code4.values())[0]['code3'].values()),
            list(list(self.code4.values())[0]['code3'].values())))
        self.assertTrue(app_code.equal_element_code([3.0, 3.0, 1.5, 1.5], [3.0, 3.0, 1.5, 1.5]))

    def test_equal_code(self):
        """

        :return:
        """
        self.assertTrue(app_code.equal_code(
            list(self.code4.items())[0], list(self.code4.items())[0]))




if __name__ == '__main__':
    unittest.main()