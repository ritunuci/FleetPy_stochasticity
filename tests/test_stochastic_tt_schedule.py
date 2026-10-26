"""The stochastic travel-time redraw schedule in `NetworkBasic`.

Travel times are redrawn at `start_time` and every `sto_tt_update_interval` seconds after it, with no
end, so the post-`end_time` tail keeps being redrawn. This replaced the explicit `sto_time_stamps` list.
Whether vehicles move on the realized draws is `stochastic_tt`'s decision alone, no longer the
schedule's.

No network is loaded: every test works on a bare `NetworkBasic` with only the attributes the method
under test reads, so the module runs in well under a second.

Run just these with:
    python -m unittest tests.test_stochastic_tt_schedule
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.misc.globals import (  # noqa: E402
    G_SIM_START_TIME,
    G_SIM_TIME_STEP,
    G_STO_TIME_STAMPS,
    G_STO_TT_UPDATE_INTERVAL,
)
from src.routing.NetworkBasic import Edge, NetworkBasic, Node  # noqa: E402

START = 25200
STEP = 10


def bare_network(**attributes):
    net = NetworkBasic.__new__(NetworkBasic)
    for name, value in attributes.items():
        setattr(net, name, value)
    return net


def configured(interval=600, start=START, step=STEP):
    net = bare_network()
    net._configure_stochastic_tt_schedule(
        {G_STO_TT_UPDATE_INTERVAL: interval, G_SIM_START_TIME: start, G_SIM_TIME_STEP: step})
    return net


class TestSchedule(unittest.TestCase):

    def test_redraws_at_start_and_every_interval_after_it(self):
        net = configured()
        on_the_grid = [t for t in range(START - 1200, 90000 + 1, STEP) if net._is_stochastic_tt_update_time(t)]
        self.assertEqual(on_the_grid, list(range(START, 90000 + 1, 600)))

    def test_it_is_open_ended_through_the_tail(self):
        net = configured()
        for t in (70200, 72000, 84000):          # past end_time 70000, inside the tail cap
            with self.subTest(t=t):
                self.assertTrue(net._is_stochastic_tt_update_time(t))

    def test_the_old_two_minute_offset_is_gone(self):
        net = configured()
        for t in (25320, 25920, 69720):
            with self.subTest(t=t):
                self.assertFalse(net._is_stochastic_tt_update_time(t))

    def test_nothing_before_start_time(self):
        net = configured()
        self.assertFalse(net._is_stochastic_tt_update_time(START - 600))

    def test_a_whole_float_interval_is_accepted(self):
        self.assertEqual(configured(interval=600.0).sto_tt_update_interval, 600)


class TestValidation(unittest.TestCase):

    def test_a_missing_interval_names_the_key_and_the_retired_list(self):
        with self.assertRaises(IOError) as cm:
            bare_network()._configure_stochastic_tt_schedule({G_SIM_START_TIME: START, G_SIM_TIME_STEP: STEP})
        self.assertIn(G_STO_TT_UPDATE_INTERVAL, str(cm.exception))
        self.assertIn(G_STO_TIME_STAMPS, str(cm.exception))

    def test_an_invalid_interval_is_refused(self):
        for bad in (0, -600, 605, 600.5, "ten minutes"):     # 605: off the 10 s time grid
            with self.subTest(interval=bad):
                with self.assertRaises(ValueError) as cm:
                    configured(interval=bad)
                self.assertIn(G_STO_TT_UPDATE_INTERVAL, str(cm.exception))


class TestUpdateNetwork(unittest.TestCase):
    """`update_network` redraws exactly on the schedule, and only with `stochastic_tt` on."""

    def network(self, stochastic_tt):
        net = configured() if stochastic_tt else bare_network()
        net.stochastic_tt = stochastic_tt
        net.travel_time_file_folders = {}
        net.cv_dict = {(0, 1): (0.3, 0.2), (1, 0): (0.3, 0.2)}
        net.hourly_tt_dict = {(o, d, f"hr_{h}"): 10.0 for (o, d) in net.cv_dict for h in range(7, 21)}
        return net

    def redraws(self, net, times):
        with mock.patch.object(net, "_draw_stochastic_tt_new", return_value=12.0), \
                mock.patch.object(net, "_set_stoch_edge_tt") as set_edge:
            for t in times:
                net.update_network(t)
        return set_edge.call_count

    def test_every_edge_is_redrawn_at_each_scheduled_time_only(self):
        net = self.network(stochastic_tt=True)
        self.assertEqual(self.redraws(net, range(START, START + 1200 + 1, STEP)), 3 * 2)   # 3 times x 2 edges

    def test_nothing_is_redrawn_with_stochastic_tt_off(self):
        self.assertEqual(self.redraws(self.network(stochastic_tt=False), [START, START + 600]), 0)


class TestMovementSource(unittest.TestCase):
    """`move_along_route` reads the realized draws when `stochastic_tt` is on, the edge's own travel
    time when it is off -- one edge of 10 s planned and 50 s realized, crossed in a 20 s step."""

    def network(self, stochastic_tt):
        origin, target = Node(0, 0, 0.0, 0.0), Node(1, 0, 1.0, 0.0)
        origin.add_next_edge_to(target, Edge((origin, target), 100.0, 10.0))
        origin.travel_infos_stochastic_to[1] = (50.0, 100.0)
        return bare_network(nodes=[origin, target], sim_time=0, stochastic_tt=stochastic_tt)

    def test_off_moves_on_the_planned_travel_time(self):
        pos, _dist, arrival, _passed, _times = self.network(False).move_along_route([1], (0, None, None), 20)
        self.assertEqual((pos, arrival), ((1, None, None), 10.0))

    def test_on_moves_on_the_realized_draw(self):
        pos, _dist, arrival, _passed, _times = self.network(True).move_along_route([1], (0, None, None), 20)
        self.assertEqual(arrival, -1)
        self.assertAlmostEqual(pos[2], 20.0 / 50.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
