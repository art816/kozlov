__author__ = 'art'

import copy
import glob
import itertools
import os
import unittest
import collections as cl
import operator

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

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

    def test_open_dir(self):
        """

        :return:
        """
        self.assertIsInstance(
            glob.glob(
                os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'алфавит', '*')),
            list)

    def test_get_code3(self):
        """

        :rtype : object
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
        code4_first_values_for_key = [list(self.code4[key]['code3'].values())[0]
                                      for key in self.code4]
        self.assertTrue(
            code4_first_values_for_key == sorted(code4_first_values_for_key,
                                                 reverse=True))

    def test_find_by_code4(self):
        """
        :return:
        """
        # TODO сделать нормальный тест, разбить на части
        import io
        from PIL import Image
        import PyPaint

        img = PyPaint.main()
        print(img.shape)
        pix_array_red = app_code.get_img_chanel(img, 'red')
        pix_array_blue = app_code.get_img_chanel(img, 'red')
        target_pix_array = app_code.select_point(pix_array_red, random=0)
        example_pix_array = app_code.select_point(pix_array_blue, random=1)

        pix_array_red = app_code.transform(np.where(pix_array_red == 1),
                                           pix_array_red.shape)
        target_pix_array = app_code.transform(np.where(target_pix_array == 1),
                                              target_pix_array.shape)
        # pix_array_blue = app_code.transform(np.where(pix_array_blue == 1),
        #                                     pix_array_blue.shape)
        # example_pix_array = app_code.transform(np.where(example_pix_array == 1),
        #                                        example_pix_array.shape)

        plt.figure(1)
        rgbArray = np.zeros(
                (target_pix_array.shape[0], target_pix_array.shape[1], 3),
                'uint8')
        rgbArray[...,0] = 255*target_pix_array
        rgbArray[...,1] = 255*pix_array_red
        plt.imshow(rgbArray, interpolation='none')
        rgbArray = np.zeros(
            (example_pix_array.shape[0], example_pix_array.shape[1], 3),
            'uint8')
        rgbArray[..., 0] = 255 * example_pix_array
        rgbArray[..., 1] = 255 * pix_array_blue
        plt.figure(2)
        plt.imshow(rgbArray, interpolation='none')
        plt.show()
        # plt.figure(1)
        # rgbArray = np.zeros(
        #     (target_pix_array.shape[0], target_pix_array.shape[1], 3),
        #     'uint8')
        # rgbArray[...,0] = 255*target_pix_array
        # rgbArray[...,2] = 255*example_pix_array
        # plt.imshow(rgbArray, interpolation='none')
        # plt.show()
        # input()
        # code3 = app_code.get_code3(
        #     np.array([self.x, self.y]))
        # example_code4 = app_code.get_code4(self.code3)
        # target_pix_array = app_code.open_image(self.image_path, rotate=0)#'C:\\Users\\art\\Documents\\MATLAB\\Apps\\козлов\\ABVG.bmp', rotate=0)
        # target_pix_array = app_code.open_image(
        #     os.path.join(os.path.dirname(os.path.abspath(__file__)),
        #                  cfg.image_names[1]), rotate=0)
        # target_code3, target_index_of1 = app_code.get_code3(
        #     target_pix_array)

        #make affine transform
        # pix_array_red = app_code.transform(np.where(pix_array_red == 1), pix_array_red.shape)
        # target_pix_array = app_code.transform(np.where(target_pix_array == 1),
        #                                        target_pix_array.shape)
        target_code3, target_index_of1 = app_code.get_code3(
            target_pix_array)
        example_code3, example_index_of1 = app_code.get_code3(
            example_pix_array)
        target_code4 = app_code.get_code4(target_code3)
        example_code4 = app_code.get_code4(example_code3)
        app_code.add_key_for_next_small4(target_code4)
        app_code.add_key_for_next_small4(example_code4)

        # TODO открывать изображения в новом потоке
        # Process(target=plt.show, args=(None,)).start()
        # target_code4 = copy.deepcopy(example_code4)
        # plt.figure(1)
        # plt.imshow(target_pix_array, interpolation='none')
        # plt.figure(2)
        # plt.imshow(example_pix_array, interpolation='none')
        # plt.show()

        pairs_equal_code = app_code.find_by_code4(target_code4, example_code4)

        if pairs_equal_code:
            # TODO помним что есть повторяющиеся пары в коде.
            # Преобразование строится методом наименьших квадратов.
            # К повторяющимся парам больше доверия
            # print(len(pairs_equal_code), pairs_equal_code)
            for pairs in pairs_equal_code:
                dict_pairs = app_code.get_dict_pairs(pairs)
                unique_pairs = app_code.get_unique_pairs_by_truth(dict_pairs)
                target = app_code.indecs_1d_to_2d(
                    [one_pair[0] for one_pair in unique_pairs], target_index_of1)
                example = app_code.indecs_1d_to_2d(
                    [one_pair[1] for one_pair in unique_pairs], example_index_of1)
                A, std = app_code.find_transform(target, example)

                # print("dict_pairs={}\tunique_t={}\tunique_e={}\tunique_pairs={}\n".format(dict_pairs, unique_target, unique_example, unique_pairs))
                print(A, std, type(std))
                # if std > cfg.std:
                #     continue

                # print(image_array.shape, np.max(image_array), image_array)
                # print(np.max(image_array), np.min(image_array), image_array.shape,
                #      width, height)

                transform_example_base = np.zeros(target_pix_array.shape, 'uint8')
                xy_base = np.dot(example, A)
                transform_example_base[
                    xy_base[:, 0].astype(int) % target_pix_array.shape[0],
                    xy_base[:, 1].astype(int) % target_pix_array.shape[1]] = 1

                x, y = np.nonzero(pix_array_blue)
                transform_pix_array_blue = np.zeros(target_pix_array.shape)
                example_one = np.vstack([x, y, np.ones(x.shape)]).T
                xy = np.dot(example_one, A)
                transform_pix_array_blue[
                    xy[:, 0].astype(int) % target_pix_array.shape[0],
                    xy[:, 1].astype(int) % target_pix_array.shape[1]] = 1

                target_base = np.zeros(target_pix_array.shape, 'uint8')
                target_base[
                    target[:, 0].astype(int) % target_pix_array.shape[0],
                    target[:, 1].astype(int) % target_pix_array.shape[1]] = 1

                rgbArray = np.ones(
                    (target_pix_array.shape[0], target_pix_array.shape[1], 3),
                    'uint8')
                rgbArray[..., 0] = 255 * (pix_array_red)# - 127 * target_base
                rgbArray[..., 1] = 255 * (transform_pix_array_blue)# - 127 * transform_example_base
                rgbArray[..., 2] = 30 * transform_example_base + 60 * target_base# np.ones(target_pix_array.shape,
                # 'uint8')
                plt.figure(3)
                plt.imshow(rgbArray, interpolation='none')
                # TODO открывать изображения в новом потоке
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
            os.path.dirname(os.path.abspath(__file__)), cfg.image_names[1]),
            rotate=0)
        target_pix_array[:,0] = 1
        target_pix_array[:,-1] = 1
        target_pix_array[0, :] = 1
        target_pix_array[-1, :] = 1
        plt.figure(1)
        plt.imshow(target_pix_array, interpolation='none')
        target_code3, target_index_of1 = app_code.get_code3(
            target_pix_array)
        transform_pix_array = app_code.transform(target_index_of1,
                                                 target_pix_array.shape)
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
        # TODO Сейчас не работает так как переупорядочевание немного другое
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
        A = cl.OrderedDict([((0, 4, 10, 16),
                        cl.OrderedDict([('code3',
                                      cl.OrderedDict(
                                          [((0, 10, 16), 1.0),
                                           ((4, 10, 16), 1.0),
                                           ((0, 4, 10), 1.0),
                                           ((0, 4, 16), 1.0)]))]))])
        B = cl.OrderedDict([((0, 10, 4, 16),
                        cl.OrderedDict([('code3',
                                      cl.OrderedDict(
                                          [((4, 10, 16), 1.0),
                                           ((4, 10, 16), 1.0),
                                           ((0, 16, 4), 1.0),
                                           ((4, 0, 10), 1.0)]))]))])
        self.assertIsNone(app_code.corr_between_points(
            list(A.items())[0], list(B.items())[0]))

    def test_equal_element_code(self):
        """

        :return:
        """
        self.assertTrue(app_code.equal_element_code(
            list(list(self.code4.values())[0]['code3'].values()),
            list(list(self.code4.values())[0]['code3'].values())))
        self.assertTrue(app_code.equal_element_code([3.0, 3.0, 1.5, 1.5],
                                                    [3.0, 3.0, 1.5, 1.5]))

    def test_equal_code(self):
        """

        :return:
        """
        self.assertTrue(app_code.equal_code(
            list(self.code4.items())[0], list(self.code4.items())[0]))

    def test_networkx(self):
        """

        :return:
        """

        def hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0,
                          xcenter=0.5):
            """
            If there is a cycle that is reachable from root, then result will not be a hierarchy.
            G: the graph
            root: the root node of current branch
            width: horizontal space allocated for this branch - avoids overlap with other branches
            vert_gap: gap between levels of hierarchy
            vert_loc: vertical location of root
            xcenter: horizontal location of root
            """

            def h_recur(G, root, width=1., vert_gap=0.2, vert_loc=0,
                        xcenter=0.5, pos=None, parent=None, parsed=None):
                if parsed is None:
                    parsed = []
                if root not in parsed:
                    parsed.append(root)
                    if pos is None:
                        pos = {root: (xcenter, vert_loc)}
                    else:
                        pos[root] = (xcenter, vert_loc)
                    neighbors = G.neighbors(root)
                    if parent is not None:
                        neighbors.remove(parent)
                    if len(neighbors) != 0:
                        dx = width / len(neighbors)
                        nextx = xcenter - width / 2 - dx / 2
                        for neighbor in neighbors:
                            nextx += dx
                            pos = h_recur(G, neighbor, width=dx,
                                          vert_gap=vert_gap,
                                          vert_loc=vert_loc - vert_gap,
                                          xcenter=nextx, pos=pos,
                                          parent=root, parsed=parsed)
                return pos

            return h_recur(G, root, width=1., vert_gap=0.2, vert_loc=0,
                           xcenter=0.5)

        G = nx.Graph()
        G_sq = nx.Graph()
        app_code.add_key_for_next_small4(self.code4)
        for element in self.code4:
            next_4 = self.code4[element]['next_4']
            G.add_node(element,
                       weight=sum(self.code4[element]['code3'].values()),
                       level=self.code4[element]['level'])
            G.add_edge(element, next_4, weight=1)
            current_code = np.array(list(self.code4[element]['code3'].values()))
            current_code = tuple(current_code / current_code[0])
            G_sq.add_node(current_code)

            if next_4 is not None:
                next_code = np.array(list(self.code4[next_4]['code3'].values()))
                next_code = tuple(next_code / next_code[0])
            else:
                next_code = None
            G_sq.add_edge(current_code, next_code, weight=1)

        # print(nx.info(G))
        # nodes_labels = dict([(u, d) for u, d in G.nodes(data=True)])
        nodes_labels = nx.get_node_attributes(G, 'level')
        # print(nodes_labels)
        # pos = nx.spring_layout(G)
        # nx.draw(G)
        pos = hierarchy_pos(G, None)  # list(self.code4.keys())[0])
        nx.draw_networkx_labels(G, pos, labels=nodes_labels)
        nx.draw(G, pos=pos, with_labels=True)
        plt.show()

        pos = hierarchy_pos(G_sq, None)  # list(self.code4.keys())[0])
        # pos = nx.spring_layout(G_sq)
        nx.draw(G_sq, pos=pos, with_labels=False)
        plt.show()

    def test_select_point(self):
        """

        :return:
        """
        app_code.select_point(self.pix_array)


if __name__ == '__main__':
    unittest.main()
