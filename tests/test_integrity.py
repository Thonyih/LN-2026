import sys, unittest, json
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from config import LABELS, FIELDS, CACHE
from data import clean, note_keys
from augment import generate
from rules import apply
from training_utils import windows, normalised_micro_loss
from models import score
from evaluation import store_scores, read_scores


class Integrity(unittest.TestCase):
    def test_reversed_label_columns_and_reordered_source_records(self):
        class ReversedClassifier:
            classes_ = np.array(list(reversed(LABELS)))

            def decision_function(self, frame):
                return np.array([np.arange(12), np.arange(12)[::-1]])

        frame = self.fixture()
        scores = score(ReversedClassifier(), frame)
        with tempfile.TemporaryDirectory(dir=CACHE) as temporary:
            path = Path(temporary) / "scores.npz"
            store_scores(path, frame, scores, {})
            ordered = read_scores(path, frame.iloc[::-1])
        predictions = np.array(LABELS)[ordered.argmax(1)]
        self.assertEqual(predictions.tolist(), ["Surgery", "Cardiovascular-Pulmonary"])

    def fixture(self):
        rows = []
        for i, text in enumerate(
            [
                "HISTORY: source one words here. IMPRESSION: skin lesion diagnosis one.",
                "HISTORY: source two words here. IMPRESSION: skin lesion diagnosis two.",
            ]
        ):
            rows.append(
                {
                    "source_record": 10 + i,
                    "medical_specialty": "Dermatology",
                    "description": "skin lesion",
                    "sample_name": "visit",
                    "transcription": text,
                    "keywords": "dermatology, skin",
                    "group": str(i),
                }
            )
        return pd.DataFrame(rows)

    def test_donor_isolation_and_original_preservation(self):
        tr = self.fixture()
        before = tr.copy(deep=True)
        aug, synth = generate(tr, "sections")
        pd.testing.assert_frame_equal(tr, before)
        self.assertGreater(len(synth), 0)
        for _, r in synth.iterrows():
            self.assertIn(r.base_source_record, {10, 11})
            self.assertTrue(set(json.loads(r.donor_source_records)) <= {10, 11})
        self.assertTrue(any(json.loads(x) for x in synth.donor_source_records))
        solo = tr.iloc[:1]
        _, one = generate(solo, "sections")
        self.assertTrue(all(not json.loads(x) for x in one.donor_source_records))

    def test_keyword_removal_does_not_erase_remaining_terms(self):
        tr = self.fixture()
        c = clean(tr, "stripped")
        self.assertEqual(c.keywords.iloc[0], "skin")
        self.assertEqual(tr.keywords.iloc[0], "dermatology, skin")

    def test_duplicate_exclusion_and_keyword_precedence(self):
        tr = self.fixture().iloc[:1]
        te = tr.copy()
        te["keywords"] = "surgery, incision"
        s = np.zeros((1, 12))
        s[0, LABELS.index("Dermatology")] = 10
        s[0, LABELS.index("Neurology")] = 9
        p, r = apply(te, s, tr, "memory")
        self.assertEqual(p[0], "Neurology")
        self.assertEqual(r[0], "memory")
        p, r = apply(te, s, tr, "both")
        self.assertEqual(p[0], "Surgery")
        self.assertEqual(r[0], "keyword")

    def test_final_partial_window(self):
        w = list(windows(np.arange(2613)))
        self.assertEqual(len(w), 164)
        self.assertEqual(len(w[-1]), 5)
        self.assertEqual(np.concatenate(w).tolist(), list(range(2613)))

    def test_weighted_microbatch_gradient_matches_effective_batch(self):
        import torch

        torch.manual_seed(4)
        x = torch.randn(5, 3)
        y = torch.tensor([0, 0, 1, 1, 1])
        weights = torch.tensor([1.0, 3.0])
        a = torch.nn.Linear(3, 2)
        b = torch.nn.Linear(3, 2)
        b.load_state_dict(a.state_dict())
        torch.nn.functional.cross_entropy(a(x), y, weight=weights).backward()
        denominator = weights[y].sum()
        normalised_micro_loss(b(x[:3]), y[:3], weights, denominator).backward()
        normalised_micro_loss(b(x[3:]), y[3:], weights, denominator).backward()
        for pa, pb in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(pa.grad, pb.grad)


if __name__ == "__main__":
    unittest.main()
