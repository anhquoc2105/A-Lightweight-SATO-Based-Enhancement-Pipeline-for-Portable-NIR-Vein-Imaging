import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from main import clear_bottom_border, process_image


PROJECT_DIR = Path(__file__).resolve().parents[1]


class VeinPipelineRegressionTests(unittest.TestCase):
    def test_clear_bottom_border_only_blacks_out_requested_rows(self):
        vein_mask = np.full((10, 8), 255, dtype=np.uint8)

        result = clear_bottom_border(vein_mask, border_rows=3)

        self.assertTrue((result[:-3] == 255).all())
        self.assertFalse(result[-3:].any())

    def test_pipeline_uses_three_pixel_local_threshold_window(self):
        with patch("main.phan_doan_lan_can", return_value=np.zeros((300, 300), dtype=np.uint8)) as threshold:
            process_image(PROJECT_DIR / "image" / "1.png")

        self.assertEqual(threshold.call_args.kwargs["ksize"], 3)


if __name__ == "__main__":
    unittest.main()
