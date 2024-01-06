class StandardScaler:
    """
    Standard scaler for input normalization
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def normalize(self, data):
        return (data - self.mean) / self.std

    def denormalize(self, data):
        return (data * self.std) + self.mean
