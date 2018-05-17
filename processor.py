import parsing
import unittest
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from scipy.ndimage.filters import gaussian_filter


class Processor():

    def __init__(self, link_file, contour_path, dicom_path):
        self.data_links = []
        self.contour_path = contour_path
        self.dicom_path = dicom_path
        self.data_links = self.get_data_links(link_file)
        self.img_width, self.img_height = None, None
        self.dataset_files = self.aggregate_dataset_files(True)

    def get_data_links(self, link_file):
        """
        Builds the data_links dictionary.
        :return: Key: Contour_id, Value: DICOM_id
        """
        df = pd.read_csv(link_file)
        d = {}
        for i in range(len(df.values)):
            d[df.loc[i, 'original_id']] = df.loc[i, 'patient_id']
        return d

    def parse_single_img_annotation(self, dicom_file, icontour_file, ocontour_file=None):
        """
        Parses a single DICOM, icontour, ocontour triple. The dicom file informs the image dimensions of the contour array.
        :param dicom_file: Path to dicom file
        :param icontour_file: Path to icontour file
        :param ocontour_file: Path to ocontour file
        :return: Tuple (dicom_arr, icontour_arr, ocontour_arr).  Ocontour returns None if no corollary ocontour exists
            Each array is of dimension (height, width)
        """

        dicom_arr = parsing.parse_dicom_file(dicom_file)
        if self.img_height is None or self.img_width is None:
            self.img_height = dicom_arr.shape[0]
            self.img_width = dicom_arr.shape[1]
        else:
            assert self.img_height == dicom_arr.shape[0], 'All DICOM img height should be the same: ' + dicom_file
            assert self.img_width == dicom_arr.shape[1], 'All DICOM img width should be the same: ' + dicom_file

        icontour_lst = parsing.parse_contour_file(icontour_file)
        assert self.img_height >= max([i[0] for i in icontour_lst]), 'Contour out of width bounds: ' + icontour_file
        assert self.img_width >= max([i[1] for i in icontour_lst]), 'Contour out of height bounds: ' + icontour_file
        icontour_arr = parsing.poly_to_mask(icontour_lst, self.img_width, self.img_height).astype(bool)

        try:
            ocontour_lst = parsing.parse_contour_file(ocontour_file)
            assert self.img_height >= max([i[0] for i in icontour_lst]), 'Contour out of width bounds: ' + ocontour_file
            assert self.img_width >= max([i[1] for i in icontour_lst]), 'Contour out of height bounds: ' + ocontour_file
            ocontour_arr = parsing.poly_to_mask(ocontour_lst, self.img_width, self.img_height).astype(bool)
        except (TypeError, FileNotFoundError):
            ocontour_arr = np.zeros_like(dicom_arr)

        return dicom_arr, icontour_arr, ocontour_arr

    def aggregate_dataset_files(self, exclude_missing_ocontour=False):
        """
        Crawls through folders and compiles a list of dicom/icontour file names that comprises the dataset.
        :param: exclude_missing_ocontour: Flag; If true, filters dataset to only include examples that include ocontour
            annotations
        :return: List of tuples: (dicom_file, icontour_file, ocontour_file)
            ocontour_file is None of the corrolary ocontour file doesn't exist
        """
        dataset_files = []
        for c_id in os.listdir(self.contour_path):
            if c_id[0] == '.':
                continue
            d_id = self.data_links[c_id]
            for ic_file in os.listdir(os.path.join(self.contour_path, c_id, 'i-contours')):
                ic_path_file = os.path.join(self.contour_path, c_id, 'i-contours', ic_file)
                rgx = re.compile('IM-(\d+)-(\d+)-icontour-manual.*')
                c_slice_id = rgx.match(ic_file)
                assert c_slice_id.group(1) is not None, "Contour file naming pattern unexpected " + ic_path_file

                oc_file = 'IM-' + c_slice_id.group(1) + '-' + c_slice_id.group(2) + '-ocontour-manual.txt'
                oc_path_file = os.path.join(self.contour_path, c_id, 'o-contours', oc_file)
                if not os.path.isfile(oc_path_file):
                    oc_path_file = None

                d_slice_id = str(int(c_slice_id.group(2)))
                d_path_file = os.path.join(self.dicom_path, d_id, d_slice_id + '.dcm')
                dataset_files.append((d_path_file, ic_path_file, oc_path_file))

        if exclude_missing_ocontour:
            dataset_files = list(filter(lambda x: x[2] is not None, dataset_files))

        return dataset_files

    def batch_generator(self, batch_sz=8):
        """
        Generator for cycling through (img, contour) batches.
        """
        l = len(self.dataset_files)
        assert l >= batch_sz
        while True:
            shuffled_idx = np.random.permutation(l)
            curr_i = 0
            if l - curr_i >= batch_sz:
                batch_img, batch_icontour, batch_ocontour = [], [], []
                for i in range(batch_sz):
                    d_file, ic_file, oc_file = self.dataset_files[shuffled_idx[curr_i + i]]
                    img, icontour, ocontour = self.parse_single_img_annotation(d_file, ic_file, oc_file)
                    batch_img.append(img)
                    batch_icontour.append(icontour)
                    batch_ocontour.append(ocontour)
                yield np.asarray(batch_img), np.asarray(batch_icontour), np.asarray(batch_ocontour)

class TestProcessor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._p = Processor(link_file=os.path.join('../final_data', 'link.csv'),
                           contour_path=os.path.join('../final_data', 'contourfiles'),
                           dicom_path=os.path.join('../final_data', 'dicoms'))

    def test_data_links(self):
        self.assertTrue(self._p.data_links['SC-HF-I-1'] == 'SCD0000101')
        self.assertTrue(self._p.data_links['SC-HF-I-2'] == 'SCD0000201')
        self.assertTrue(self._p.data_links['SC-HF-I-4'] == 'SCD0000301')
        self.assertTrue(self._p.data_links['SC-HF-I-5'] == 'SCD0000401')
        self.assertTrue(self._p.data_links['SC-HF-I-6'] == 'SCD0000501')
        self.assertTrue(len(self._p.data_links) == 5)

    def vis_batch_img_contour(self, image_batch, *overlay_batches):
        """
        Visualizes a batch of images and contours and their overlay for visual validation
        Input, variable length list of array batches to visualize.
        The first argument is the input image, used as a background.
        The rest of the arguments contain icontour/ocontour/prediction which we overlay on top of the image.
        """
        l = image_batch.shape[0]
        k = len(overlay_batches)
        m = max(k, 1)
        for i in range(k):
            assert overlay_batches[i].shape[0] == image_batch.shape[0]

        plt.figure(figsize=(5 * m, 5 * l))
        for i in range(l):
            dicom_arr = image_batch[i]
            plt.subplot(l, m, int(m * i + 1))
            plt.imshow(dicom_arr, cmap='copper', alpha=0.7)

            for j in range(k):
                contour_arr = overlay_batches[j][i]
                plt.subplot(l, m, int(m * i + j + 1))
                plt.imshow(dicom_arr, cmap='copper', alpha=0.7)
                plt.imshow(contour_arr, cmap='jet', alpha=0.5)
        plt.show()

    def test_vis_dicom_parse(self):
        """
        This test doesn't conventionally unittest.  Instead it visualizes a few contour/dicom examples
        """
        icontour_path = os.path.join(self._p.contour_path, 'SC-HF-I-1', 'i-contours')
        icontour_files = ['IM-0001-0048-icontour-manual.txt', 'IM-0001-0059-icontour-manual.txt', 'IM-0001-0068-icontour-manual.txt']
        ocontour_path = os.path.join(self._p.contour_path, 'SC-HF-I-1', 'o-contours')
        ocontour_files = [None, 'IM-0001-0059-ocontour-manual.txt', None]
        dicom_path = os.path.join(self._p.dicom_path, 'SCD0000101')
        dicom_files = ['48.dcm', '59.dcm', '68.dcm']
        l = len(dicom_files)
        plt.figure(figsize=(5 * 3, 5 * l))
        img_batch, icontour_batch, ocontour_batch = [], [], []
        for c, o, d in zip(icontour_files, ocontour_files, dicom_files):
            c = os.path.join(icontour_path, c)
            if o is not None:
                o = os.path.join(icontour_path, o)
            d = os.path.join(dicom_path, d)
            dicom_arr, icontour_arr, ocontour_arr = self._p.parse_single_img_annotation(d, c, o)
            self.assertTrue(dicom_arr.shape == icontour_arr.shape)
            self.assertTrue((ocontour_arr is None) or (dicom_arr.shape == ocontour_arr.shape))
            img_batch.append(dicom_arr)
            icontour_batch.append(icontour_arr)
            ocontour_batch.append(ocontour_arr)
        img_batch = np.asarray(img_batch)
        icontour_batch = np.asarray(icontour_batch)
        self.vis_batch_img_contour(img_batch, icontour_batch)

    def test_aggregate_dataset_files(self):
        d_file1 = os.path.join(self._p.dicom_path, 'SCD0000101', '48.dcm')
        ic_file1 = os.path.join(self._p.contour_path, 'SC-HF-I-1', 'i-contours', 'IM-0001-0048-icontour-manual.txt')
        oc_file1 = None

        d_file2 = os.path.join(self._p.dicom_path, 'SCD0000501', '219.dcm')
        ic_file2 = os.path.join(self._p.contour_path, 'SC-HF-I-6', 'i-contours', 'IM-0001-0219-icontour-manual.txt')
        oc_file2 = os.path.join(self._p.contour_path, 'SC-HF-I-6', 'o-contours', 'IM-0001-0219-ocontour-manual.txt')

        dataset_lst = self._p.aggregate_dataset_files(False)
        self.assertTrue((d_file1, ic_file1, oc_file1) in dataset_lst)
        self.assertTrue((d_file2, ic_file2, oc_file2) in dataset_lst)

        dataset_lst1 = self._p.aggregate_dataset_files(True)
        self.assertTrue((d_file1, ic_file1, oc_file1) not in dataset_lst1)
        self.assertTrue((d_file2, ic_file2, oc_file2) in dataset_lst1)


    def test_batch_gen(self):
        '''
        Here we visually validate that the generator is randomly shuffling with each epoch as appropriate and that
        each image/contour batch corresponds accordingly. Helpful to zoom in with larger batch_sizes.
        '''
        batch_sz = 3
        nb_iter = 4
        self._p.dataset_files = self._p.dataset_files[:batch_sz * 2]
        g = self._p.batch_generator(batch_sz=batch_sz)
        img_batch, icontour_batch, ocontour_batch = [], [], []
        for i in range(nb_iter):
            batch_d, batch_ic, batch_oc = next(g)
            self.assertEqual(batch_d.shape, batch_ic.shape)
            self.assertEqual(batch_d.shape, batch_oc.shape)
            img_batch.append(batch_d)
            icontour_batch.append(batch_ic)
            ocontour_batch.append(batch_oc)
        img_batch = np.concatenate(img_batch, axis=0)
        icontour_batch = np.concatenate(icontour_batch, axis=0)
        ocontour_batch = np.concatenate(ocontour_batch, axis=0)
        self.vis_batch_img_contour(img_batch, icontour_batch, ocontour_batch)


    def test_icontour_heuristic(self):
        """
        This test finds a heuristic threshold for labeling icontour, given an ocontour label and image.
        To do so, choose the threshold that maximizes dice on the train set and report performance on the test set.
        Using this method, we find 0.35% of max pixel value to be the optimal threshold, resulting in 0.82 dice coef.
        """
        def dice_coef(x, y):
            assert(len(x.shape) == 3)
            return np.mean(2 * np.sum(x * y, axis=(1,2)) / (np.sum(x, axis=(1,2)) + np.sum(y, axis=(1,2))), axis=0)

        a1 = np.array([[[1, 1, 1, 0, 0]]])
        a2 = np.array([[[1, 1, 1, 0, 0]]])
        a3 = np.array([[[0, 0, 0, 1, 1]]])
        a4 = np.array([[[0, 0, 1, 1, 1]]])
        a5 = np.concatenate([a2, a3, a4], axis=0)
        a6 = np.concatenate([a1, a1, a1], axis=0)

        self.assertEqual(dice_coef(a1, a2), 1)
        self.assertEqual(dice_coef(a1, a3), 0)
        self.assertEqual(dice_coef(a1, a4), 1. / 3)
        self.assertEqual(dice_coef(a5, a6), 4. / 9)

        g = self._p.batch_generator(batch_sz=int(0.5 * len(self._p.dataset_files)))
        for phase in ['train', 'test']: #First half of dataset for "training", second half for "testing"
            img_batch, icontour_batch, ocontour_batch = next(g)
            for pc in [x / 100 for x in range(0, 100, 5)]:
                iheur = np.zeros_like(img_batch)
                for i in range(iheur.shape[0]):
                    #Threshold hold is relative to the max value of the image within the region of the ocontour.
                    iheur[i][(img_batch[i] > pc * np.amax(img_batch[i][ocontour_batch[i] == 1])) & (ocontour_batch[i] == 1)] = 1
                print(phase, "threshold:", pc, "; dice:", dice_coef(icontour_batch, iheur))

    def test_vis_optimal_icontour_heuristic(self):
        """
        Visualize a few examples given the 0.35 threshold icontour heuristic found above (test_icontour_heuristic).
        """
        g = self._p.batch_generator(batch_sz=3)
        img_batch, icontour_batch, ocontour_batch = next(g)
        iheur = np.zeros_like(img_batch)
        for i in range(iheur.shape[0]):
            iheur[i][(img_batch[i] > 0.35 * np.amax(img_batch[i][ocontour_batch[i] == 1])) & (ocontour_batch[i] == 1)] = 1
        self.vis_batch_img_contour(img_batch, ocontour_batch, icontour_batch, iheur)

if __name__ == '__main__':
    unittest.main()
