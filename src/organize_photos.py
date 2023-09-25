import os
import argparse
import glob
import logging
import coloredlogs
from PIL import Image
from PIL.ExifTags import TAGS
import polars as pl
import pandas as pdimport yaml

coloredlogs.install(level='INFO',fmt='%(asctime)s %(levelname)s %(message)s')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--org_cfg', type=str, default='org_cfg.yaml', help='configuration for organizing images')

    args = parser.parse_args()
    return args
    

class FotoFinesseProOrganizer:
    def __init__(self, source, output):
        logger = logging.getLogger(__name__)
        self.source = source
        self.output = output
        self.img_filepaths = self.get_img_filepaths()
        logger.info('Found {} images'.format(len(self.img_filepaths)))   
    
    def get_img_filepaths(self):
        """
        get list of all image filepaths in source directory
        recursive search in subdirectories
        supported formats: png, jpg, jpeg
        source: directory containing images
        """
        source = self.source
        if os.path.exists(source):
            img_filepaths = glob.glob(source+'/*.png', recursive=True)
            img_filepaths.extend(glob.glob(source+'/*.jpg', recursive=True))
            img_filepaths.extend(glob.glob(source+'/*.jpeg', recursive=True))
            if not img_filepaths:
                raise ValueError('No images found in {}'.format(source))
        else:
            raise ValueError('{} does not exist'.format(source))
        return img_filepaths
    
    def get_metadata(self, images, save=False):
        metadata = []
        for img in images:
            img_metadata = {}
            exifdata = img.getexif()
            for tag_id in exifdata:
                tag = TAGS.get(tag_id, tag_id)
                data = exifdata.get(tag_id)
                if isinstance(data, bytes):
                    data = data.decode()
                img_metadata[tag] = data
            metadata.append(img_metadata)
        metadata = pd.DataFrame(metadata)
        if save:
            metadata.to_csv('metadata.csv')
        return metadata

    def read_images(self, filepaths):
        filepaths = [f for f in filepaths if os.path.exists(f)]
        if not filepaths:
            raise ValueError('No filepaths exists {}'.format(','.join(filepaths)))
        images = [Image.open(filepath) for filepath in filepaths]
        return images


    def parse_date(self, date):
        """
        parse date from metadata
        date: date in string format
        """
        logger = logging.getLogger(__name__)
        logger.info('date {}'.format(date))
        try:
            date = date.split(' ')[0]
            year, month, day = date.split(':')
            return day, month, year
        except:
            return 'unk','unk','unk'
    
    def organize_by_date(self, metadata):
        """
        organize images by date
        metadata: metadata of images
        output_dir: directory to store organized images
        """
        logger = logging.getLogger(__name__)
        for _, row in metadata.iterrows():
            day, month, year = self.parse_date(row['DateTime'])
            out_dir = os.path.join(self.output, year, month, day)
            logger.info('out_dir {}'.format(out_dir))
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
                logger.info('Created directory {}'.format(out_dir))
            img = Image.open(row['filepath'])
            logger.info('Saving  {}'.format(os.path.join(out_dir, row['filepath'].split('/')[-1])))
            img.save(os.path.join(out_dir, row['filepath'].split('/')[-1]))

    
    def organize(self, org_cfg):
        """
        input_dir: directory containing images  
        metadata: metadata of images
        output_dir: directory to store organized images
        org_cfg: configuration for organizing images (dict)
        """
        img_filepaths = self.img_filepaths
        images = self.read_images(img_filepaths)
        metadata = self.get_metadata(images)
        metadata['filepath'] = img_filepaths
        if org_cfg['by'] == 'date':
            self.organize_by_date(metadata)
        



def main():
    args = parse_args()
    logger = logging.getLogger(__name__)
    organizer = FotoFinesseProOrganizer(args.source, args.output)
    org_cfg =  {'by':'date'}
    organizer.organize(org_cfg)


if __name__=="__main__":
    main()
        