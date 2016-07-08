__author__ = 'art'

import unittest
import numpy as np
import itertools

import open_image


class Test(unittest.TestCase):
    def setUp(self):
        """
        :return:
        """
        self.x = np.array([0, 0, 1, 1])
        self.y = np.array([0, 1, 1, 0])
        self.image_path = 'C:\\Users\\art\\Documents\\MATLAB\\Apps\\козлов\\a_s.bmp'
        self.pix_array = open_image.open_image(self.image_path)

    def tearDown(self):
        """

        :return:
        """
        pass

    def test_polyarea(self):
        """

        :return:
        """

        self.assertEqual(open_image.polyarea(self.x, self.y), 1)

    def test_open_image(self):
        """

        :return:
        """
        self.assertIsNotNone(open_image.open_image(self.image_path))


    def test_get_code3(self):
        """

        :return:
        """
        code3 = open_image.get_code3(
            np.array([self.x, self.y]))
        self.assertEqual(len(code3.keys()), 4)
        for key in itertools.combinations(list(range(4)), 3):
            self.assertEqual(code3.get(key), 0.5)

    def test_get_code4(self):
        """

        :return:
        """
        code3 = open_image.get_code3(
            np.array([self.x, self.y]))
        code4 = open_image.get_code4(code3)
        self.assertEqual(code4.get((0, 1, 2, 3))['code3'], code3)


if __name__ == '__main__':
    unittest.main()