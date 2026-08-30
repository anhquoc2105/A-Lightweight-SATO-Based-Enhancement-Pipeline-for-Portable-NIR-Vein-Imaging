import unittest

import numpy as np

from main import clear_bottom_border


class VeinPipelineRegressionTests(unittest.TestCase):
    def test_clear_bottom_border_only_blacks_out_requested_rows(self):
        vein_mask = np.full((10, 8), 255, dtype=np.uint8)

        result = clear_bottom_border(vein_mask, border_rows=3)

        self.assertTrue((result[:-3] == 255).all())
        self.assertFalse(result[-3:].any())


if __name__ == "__main__":
    unittest.main()
