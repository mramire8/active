__author__ = 'mramire8'

from baseexpert import BaseExpert


class FunctionBasedExpert(BaseExpert):
    # This expert predicts the label based on a linear function(or logarithmic that
    # is given to the constructor?
    def __init__(self):
        self.label_instance = self.apply_model
        self.cost = self.cost_labeling

        pass

    def apply_model(self, instances=None):
        #toss coins to apply label
        pass

    def cost_labeling(self, instances=None):
        # apply cost fnction to determine cost
        pass