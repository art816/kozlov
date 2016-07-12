__author__ = 'art'

import unittest
import numpy as np
import itertools
import copy
from PIL import Image
import matplotlib.pyplot as plt

import app_code


class Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        :return:
        """
        cls.x = np.array([0, 0, 1, 1])
        cls.y = np.array([0, 1, 1, 0])
        cls.image_path = 'C:\\Users\\art\\Documents\\MATLAB\\Apps\\козлов\\a_s.bmp'
        cls.pix_array = app_code.open_image(cls.image_path)
        cls.code3 = app_code.get_code3(
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
        code3 = app_code.get_code3(
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
        code3 = app_code.get_code3(
             np.array([self.x, self.y]))
        example_code4 = app_code.get_code4(self.code3)

        target_pix_array = app_code.open_image(self.image_path, rotate=1)#'C:\\Users\\art\\Documents\\MATLAB\\Apps\\козлов\\ABVG.bmp', rotate=0)
        # target_pix_array = app_code.open_image('C:\\Users\\art\\Documents\\MATLAB\\Apps\\козлов\\a_s1.bmp', rotate=1)
        plt.figure(1)
        plt.imshow(target_pix_array)
        plt.figure(2)
        plt.imshow(self.pix_array)
        target_code3 = app_code.get_code3(
            target_pix_array)
        target_code4 = app_code.get_code4(target_code3)
        print('get_code4')
        app_code.add_key_for_next_small4(target_code4)
        print('add_target_code4')
        app_code.add_key_for_next_small4(example_code4)
        print('add_example_code4')
        plt.show()
        # target_code4 = copy.deepcopy(example_code4)
        self.assertEqual(*app_code.find_by_code4(target_code4, example_code4))

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