import unittest
import panda as pd

from featuregen import GroupedVariableTransformation, exceptions
from .base import BaseTestCase


class GroupedVariableTransformationTest(BaseTestCase):
    """Testing operation of the ExampleModule class"""

    def test_init_example_module(self):
        """Ensures that the twine class can be instantiated with a file"""
        # test_data_file = self.path + "test_data/.json"
        df = pd.DataFrame(
            {
                "attribute": ["A", "A", "A", "A", "A", "A", "B", "B", "B", "B", "B", "B"],
                "value": [1, 2, 4, 5, 3, 6, 100, 33, 44, 77, 77, 99],
            }
        )
        gvt = GroupedVariableTransformation(key="attribute", target="value")
        gvt.fit(df)
        print(gvt)
        gvt.transform(df)


if __name__ == "__main__":
    unittest.main()
    gvt_test = GroupedVariableTransformationTest()
