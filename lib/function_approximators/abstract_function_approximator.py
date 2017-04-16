from ..abstract_object import AbstractObject


class AbstractFunctionApproximator(AbstractObject):
    """
    Abstract class for function approximators. Only required method is predict.
    """

    def __init__(self):
        super(AbstractFunctionApproximator, self).__init__()

    def _key(self):
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        """
        Use the function approximator to predict the outcome of a given query point  
        """
        raise NotImplementedError
