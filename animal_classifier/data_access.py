from fastai.vision.all import DataBlock, ImageBlock, CategoryBlock, get_image_files, RandomSplitter, \
    parent_label, Resize


class DDGImageDAO(object):
    """
    Wrapper data access object around DataBlock API for DDG images
    """

    def __init__(self, source):
        self.source = source

    def get_data_loader(self,
                        blocks=(ImageBlock, CategoryBlock),
                        get_items=get_image_files,
                        splitter=RandomSplitter(valid_pct=0.2, seed=10),
                        get_y=parent_label,
                        item_tfms=None,
                        bs=32):
        """
        :param blocks: Tuple defining input and output types of model
        :param get_items: Function to get data to input into model
        :param splitter: Splitter type for train/validation split
        :param get_y: Function to get category labels for input data
        :param item_tfms: Transformations to apply to input data
        :param bs: Batch size as an integer
        :return dls: DataLoaders object
        """
        if item_tfms is None:
            item_tfms = [Resize(192, method='squish')]
        dls = DataBlock(
            blocks=blocks,
            get_items=get_items,
            splitter=splitter,
            get_y=get_y,
            item_tfms=item_tfms
        ).dataloaders(self.source, bs=bs)

        return dls


if __name__ == '__main__':
    SOURCE = 'animal_pictures'
    dao = DDGImageDAO(source=SOURCE)
    loader = dao.get_data_loader()
