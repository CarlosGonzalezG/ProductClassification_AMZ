import pickle
from sklearn.preprocessing import LabelEncoder

class LabelEncoderWrapper:
    """
    Class to wrap a LabelEncoder and provide a save/load method.
    """
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.is_fitted = False

    def fit(self, y):
        """
        Adjusts the encoder to the given labels.
        """
        self.label_encoder.fit(y)
        self.is_fitted = True

    def transform(self, y):
        """
        Transform labels to numerical values.
        """
        if not self.is_fitted:
            raise ValueError("LabelEncoderWrapper must be fitted before transforming data.")
        return self.label_encoder.transform(y)

    def inverse_transform(self, y):
        """
        Decode numerical values to labels.
        """
        if not self.is_fitted:
            raise ValueError("LabelEncoderWrapper must be fitted before inverse transforming data.")
        return self.label_encoder.inverse_transform(y)

    def save(self, file_path):
        """
        Save the encoder to a file.
        """
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file_path):
        """
        Load a LabelEncoderWrapper from a file.
        """
        with open(file_path, "rb") as f:
            return pickle.load(f)
