import tempfile
import unittest
from pathlib import Path

from src.argparse_util import ArgumentParser
from src.data import Data
from src.eval import _chk_route_list, evaluate_route_batch


INSTANCE_TEXT = """NAME: tiny_cuda
TYPE: VRPSPDTW
DIMENSION: 4
VEHICLES: 2
DISPATCHINGCOST: 2000
UNITCOST: 1
CAPACITY: 10
EDGE_WEIGHT_TYPE: EXPLICIT
NODE_SECTION
0,0,0,0,100,0
1,2,1,0,100,0
2,1,2,0,100,0
3,1,1,0,100,0
DISTANCETIME_SECTION
0,0,0,0
0,1,1,1
0,2,2,2
0,3,3,3
1,0,1,1
1,1,0,0
1,2,1,1
1,3,2,2
2,0,2,2
2,1,1,1
2,2,0,0
2,3,1,1
3,0,3,3
3,1,2,2
3,2,1,1
3,3,0,0
DEPOT_SECTION
0
"""


def build_data(tmp_dir: Path, backend: str = "auto") -> Data:
    instance = tmp_dir / "tiny.vrpsdptw"
    instance.write_text(INSTANCE_TEXT, encoding="utf-8")

    parser = ArgumentParser()
    parser.add_argument("--problem", 1, False)
    parser.add_argument("--compute_backend", 1)
    parser.parse(["test", "--problem", str(instance), "--compute_backend", backend])
    return Data(parser)


class ComputeBackendTest(unittest.TestCase):
    def test_batch_route_evaluation_matches_single_route_checks(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            data = build_data(Path(tmp_dir), backend="auto")

            routes = [
                [data.DC, 1, data.DC],
                [data.DC, 1, 2, 3, data.DC],
                [1, 2, 3, data.DC],
            ]

            batch = evaluate_route_batch(routes, data)
            single = [_chk_route_list(route, data) for route in routes]
            self.assertEqual(batch, single)

            self.assertTrue(batch[0][0])
            self.assertAlmostEqual(batch[0][1], 2002.0)
            self.assertTrue(batch[1][0])
            self.assertAlmostEqual(batch[1][1], 2006.0)
            self.assertFalse(batch[2][0])

    def test_cuda_request_falls_back_cleanly_when_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            data = build_data(Path(tmp_dir), backend="cuda")
            self.assertIn(data.backend.name, {"cpu", "cuda"})
            self.assertEqual(data.parallel_workers, 1 if data.backend.is_cuda else data.parallel_workers)
            route = [data.DC, 1, 2, 3, data.DC]
            self.assertEqual(_chk_route_list(route, data), data.backend.evaluate_route(route))


if __name__ == "__main__":
    unittest.main()
