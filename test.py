__author__ = 'art'

import unittest
import numpy as np
import itertools
import copy

import app_code


class Test(unittest.TestCase):
    def setUp(self):
        """
        :return:
        """
        self.x = np.array([0, 0, 1, 1])
        self.y = np.array([0, 1, 1, 0])
        self.image_path = 'C:\\Users\\art\\Documents\\MATLAB\\Apps\\козлов\\a_s.bmp'
        self.pix_array = app_code.open_image(self.image_path)

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
        code3 = app_code.get_code3(
            #np.array([self.x, self.y]))
            self.pix_array)
        code4 = app_code.get_code4(code3)
        print(code4)
        first_key = list(code4.keys())[0]
        # self.assertEqual(code4.get(first_key)['code3'], code3)
        code4_values = list(code4.get(first_key)['code3'].values())
        self.assertEqual(
            code4_values, sorted(code4_values, reverse=True))
        code4_first_values_for_key = [list(code4[key]['code3'].values())[0] for key in code4]
        self.assertEqual(
            code4_first_values_for_key, sorted(code4_first_values_for_key, reverse=True))

    def test_find_by_code4(self):
        """

        :return:
        """
        code3 = app_code.get_code3(
            np.array([self.x, self.y]))
        example_code4 = app_code.get_code4(code3)
        etalon_code4 = copy.deepcopy(example_code4)
        self.assertIsNotNone(app_code.find_by_code4(example_code4, etalon_code4))

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
        code3 = app_code.get_code3(
            self.pix_array)
        code4 = app_code.get_code4(code3)
        app_code.add_key_for_next_small4(code4)
        for key in code4:
            if code4[key]['next_4']:
                self.assertGreaterEqual(
                    list(code4[key]['code3'].values())[0],
                    list(code4[code4[key]['next_4']]['code3'].values())[0])



if __name__ == '__main__':
    unittest.main()