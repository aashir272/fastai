from fastai.vision.all import resnet18, vision_learner, error_rate, PILImage
from data_access import DAO, DDGImageDAO
from typing import Callable


class ImageModelBuilder(object):

    def __init__(self,
                 dao: DAO,
                 model: Callable,
                 metric: Callable):
        self.dao = dao
        self.model = model
        self.metric = metric

    def build_model(self, fine_tune=3):
        """
        :param fine_tune: Number of epochs to fine tune input model
        :return learner: Learner object
        """
        if not hasattr(self, 'loader'):
            # set num workers to zero for Windows
            setattr(self, 'loader', self.dao.get_data_loader(num_workers=0))
        learner = vision_learner(self.loader, self.model, self.metric)
        learner.fine_tune(fine_tune)
        return learner


if __name__ == '__main__':
    # define inputs
    SOURCE = 'animal_pictures'
    dao = DDGImageDAO(SOURCE)

    # build learner
    builder = ImageModelBuilder(dao=dao, model=resnet18, metric=error_rate)
    learner = builder.build_model()

    # test on known image
    test_path = 'animal_pictures/dog/test_dog.jpg'
    is_dog, _, prob = learner.predict(PILImage.create(test_path))
    print(f"This is a {is_dog}.")
    print(f"Probability it's a dog is:  {prob[1]:.4f}")
