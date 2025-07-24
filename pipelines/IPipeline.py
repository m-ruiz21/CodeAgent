from abc import ABC

class IPipeline(ABC):
    """
    Interface for a pipeline that can load data and poll for updates.
    """

    def load(self):
        """
        Load data from the specified URL with optional file filtering.
        """
        pass

    def poll(self):
        """
        Poll for updates in the data source and update the index if necessary.
        """
        pass