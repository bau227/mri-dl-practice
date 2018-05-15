import parsing
import unittest
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

class Processor():

    def __init__(self, link_file, contour_path, dicom_path):
        self.data_links = []
        self.contour_path = contour_path
        self.dicom_path = dicom_path
        self.data_links = self.get_data_links(link_file)
        self.img_width, self.img_height = None, None
        self.dataset_files = self.aggregate_dataset_files()

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

    def parse_dicom_icontour_pair(self, dicom_file, icontour_file):
        """
        Parses a single DICOM and icontour pair. The dicom file informs the image dimensions of the contour array.
        :param dicom_file: Path to dicom file
        :param icontour_file: Path to icontour file
        :return: Tuple (dicom_arr, icontour_arr), each array is of dimension (height, width)
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

        return dicom_arr, icontour_arr

    def aggregate_dataset_files(self):
        """
        Crawls through folders and compiles a list of dicom/icontour file names that comprises the dataset.
        :return: List of tuples: (dicom_file, icontour_file)
        """
        dataset_files = []
        for c_id in os.listdir(self.contour_path):
            if c_id[0] == '.':
                continue
            d_id = self.data_links[c_id]
            for c_file in os.listdir(os.path.join(self.contour_path, c_id, 'i-contours')):
                c_path_file = os.path.join(self.contour_path, c_id, 'i-contours', c_file)
                rgx = re.compile('IM-\d+-(\d+)-icontour-manual.*')
                c_slice_id = rgx.match(c_file)
                assert c_slice_id.group(1) is not None, "Contour file naming pattern unexpected " + c_path_file
                d_slice_id = str(int(c_slice_id.group(1)))
                d_path_file = os.path.join(self.dicom_path, d_id, d_slice_id + '.dcm')
                dataset_files.append((d_path_file, c_path_file))
        return dataset_files

    def batch_generator(self, batch_sz=8):
        '''
        Generator for cycling through (img, contour) batches.
        '''
        l = len(self.dataset_files)
        assert l >= batch_sz
        while True:
            shuffled_idx = np.random.permutation(l)
            curr_i = 0
            if l - curr_i >= batch_sz:
                batch_img = []
                batch_contour = []
                for i in range(batch_sz):
                    d_file, c_file = self.dataset_files[shuffled_idx[curr_i + i]]
                    img, contour = self.parse_dicom_icontour_pair(d_file, c_file)
                    batch_img.append(img)
                    batch_contour.append(contour)

                yield np.asarray(batch_img), np.asarray(batch_contour)

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

    def vis_batch_img_contour(self, img_batch, contour_batch):
        '''
        Visualizes a batch of images and contours and their overlay for visual validation
        '''
        l = img_batch.shape[0]
        assert img_batch.shape[0] == contour_batch.shape[0]
        plt.figure(figsize=(5 * 3, 5 * l))
        for i in range(l):
            dicom_arr = img_batch[i]
            icontour_arr = contour_batch[i]

            plt.subplot(l, 3, int(3 * i + 1))
            plt.imshow(dicom_arr, cmap='copper')
            plt.subplot(l, 3, int(3 * i + 2))
            plt.imshow(icontour_arr, cmap='jet')
            plt.subplot(l, 3, int(3 * i + 3))
            plt.imshow(dicom_arr, cmap='copper', alpha=0.7)
            plt.imshow(icontour_arr, cmap='jet', alpha=0.5)
        plt.show()

    def test_vis_dicom_parse(self):
        '''
        This test doesn't conventionally unittest.  Instead it visualizes a few contour/dicom examples
        '''
        icontour_path = os.path.join(self._p.contour_path, 'SC-HF-I-1', 'i-contours')
        icontour_files = ['IM-0001-0048-icontour-manual.txt', 'IM-0001-0059-icontour-manual.txt', 'IM-0001-0068-icontour-manual.txt']
        dicom_path = os.path.join(self._p.dicom_path, 'SCD0000101')
        dicom_files = ['48.dcm', '59.dcm', '68.dcm']
        l = len(dicom_files)
        plt.figure(figsize=(5 * 3, 5 * l))
        img_batch = []
        contour_batch = []
        for c, d in zip(icontour_files, dicom_files):
            c = os.path.join(icontour_path, c)
            d = os.path.join(dicom_path, d)
            dicom_arr, icontour_arr = self._p.parse_dicom_icontour_pair(d, c)
            self.assertTrue(dicom_arr.shape == icontour_arr.shape)
            img_batch.append(dicom_arr)
            contour_batch.append(icontour_arr)
        img_batch = np.asarray(img_batch)
        contour_batch = np.asarray(contour_batch)
        self.vis_batch_img_contour(img_batch, contour_batch)

    def test_aggregate_dataset_files(self):
        d_file = os.path.join(self._p.dicom_path, 'SCD0000101', '48.dcm')
        c_file = os.path.join(self._p.contour_path, 'SC-HF-I-1', 'i-contours', 'IM-0001-0048-icontour-manual.txt')
        dataset_lst = self._p.aggregate_dataset_files()
        self.assertTrue((d_file, c_file) in dataset_lst)

    def test_batch_gen(self):
        '''
        Here we visually validate that the generator is randomly shuffling with each epoch as appropriate and that
        each image/contour batch corresponds accordingly. Helpful to zoom in with larger batch_sizes.
        '''
        batch_sz = 3
        nb_iter = 4
        self._p.dataset_files = self._p.dataset_files[:batch_sz * 2]
        g = self._p.batch_generator(batch_sz=batch_sz)
        img_batch = []
        contour_batch = []
        for i in range(nb_iter):
            batch_d, batch_c = next(g)
            self.assertEqual(batch_d.shape, batch_c.shape)
            img_batch.append(batch_d)
            contour_batch.append(batch_c)
        img_batch = np.concatenate(img_batch, axis=0)
        contour_batch = np.concatenate(contour_batch, axis=0)
        self.vis_batch_img_contour(img_batch, contour_batch)


if __name__ == '__main__':
    unittest.main()
