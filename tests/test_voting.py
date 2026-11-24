import unittest
from intraday.voting import BOTS, fresh_record, freeze, settle


class VotingTests(unittest.TestCase):
    def test_counterfactual_and_next_session_weights(self):
        history = fresh_record()
        original = dict(zip(BOTS, [1, 1, -1, 0]))
        first = freeze(original, history)
        self.assertEqual(first["selected"], 1)
        result = settle(first, 100, 98, 5, history)
        bots = result["bots"]
        self.assertAlmostEqual(result["selected_net_usd"], -2.099)
        self.assertAlmostEqual(bots[BOTS[2]]["net_usd"], 1.901)
        self.assertAlmostEqual(bots[BOTS[2]]["delta_usd"], 4.0)
        self.assertEqual(bots[BOTS[3]]["after"], 1.0)
        self.assertLess(bots[BOTS[0]]["after"], 1)
        self.assertGreater(bots[BOTS[2]]["after"], 1)
        self.assertEqual(first["weights"], dict.fromkeys(BOTS, 1))
        tied = dict(zip(BOTS, [1, 0, -1, 0]))
        second = freeze(tied, history)
        self.assertEqual((second["selected"], second["equal"]), (-1, 0))

    def test_unprofitable_calls_on_both_sides(self):
        history = fresh_record()
        first = freeze(dict(zip(BOTS, [1, -1, 0, 0])), history)
        result = settle(first, 100, 100.01, 5, history)
        self.assertEqual(first["selected"], 0)
        self.assertEqual([r["grade"] for r in result["bots"].values()], [-1, -1, 0, 0])
