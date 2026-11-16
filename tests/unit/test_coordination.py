"""SimulationCoordinator scheduling semantics with a deterministic stand-in driver.

This is a MOCKED unit test of the threading/coalescing logic only; the same
coordinator against real ManiSkill is exercised in tests/integration and tests/gui.
"""
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from parttwin1d.sim.state import StateSnapshot
from parttwin1d.viewer.coordination import CommandQueueFull, SimulationCoordinator


class FakeDriver:
    def __init__(self, delay=0.0):
        self.joints = [SimpleNamespace(index=0, name="door_hinge", type="revolute", unit="rad", lower=0.0,
                                       upper=1.5, bounded=True, child_link="door_link")]
        self.joint_index = {"door_hinge": 0}
        self.dof = 1
        self.q = np.zeros(1)
        self.q0 = np.zeros(1)
        self.sim_time = 0.0
        self.revision = 0
        self.applied = []
        self.delay = delay
        self.closed = False
        self.thread = None
        self.env = SimpleNamespace(control_timestep=0.02)

    def _own(self):
        if self.thread is None:
            self.thread = threading.get_ident()
        assert threading.get_ident() == self.thread, "driver touched from a non-owner thread"

    def get_q(self):
        self._own()
        return self.q.copy()

    def set_q_kinematic(self, q):
        self._own()
        time.sleep(self.delay)
        q = np.asarray(q, float)
        if not (0 <= q[0] <= 1.5):
            raise ValueError("outside URDF limits")
        self.q = q
        self.applied.append(float(q[0]))

    def set_joint_kinematic(self, name, v):
        q = self.get_q()
        q[0] = v
        self.set_q_kinematic(q)

    def reset_to_q0(self):
        self.set_q_kinematic(self.q0)

    def step_dynamics(self, n, drive_target=None, joint_force=None):
        self._own()
        self.sim_time += 0.02 * n
        if drive_target is not None:
            self.q = self.q + 0.5 * (np.asarray(drive_target) - self.q)

    def snapshot(self, mode, note="", request_id=-1):
        self._own()
        self.revision += 1
        q = self.q.copy()
        q.flags.writeable = False
        return StateSnapshot(self.revision, self.sim_time, time.time(), 0, mode, ("door_hinge",), q, q * 0,
                             np.array([[0, 1.5]]), {}, (), note, request_id)

    def close(self):
        self._own()
        self.closed = True


@pytest.fixture()
def coord():
    holder = {}

    def factory():
        holder["d"] = FakeDriver(delay=0.002)
        return holder["d"]
    c = SimulationCoordinator(factory, playback_hz=100).start()
    c.fake = holder["d"]
    yield c
    c.stop()


def test_initial_snapshot_and_immutability(coord):
    s = coord.snapshot()
    assert s.revision == 1 and s.mode == "kinematic"
    with pytest.raises(ValueError):
        s.q[0] = 3.0


def test_slider_burst_is_coalesced_and_latest_wins(coord):
    rids = [coord.submit_joint("door_hinge", v) for v in np.linspace(0, 1.0, 300)]
    s = coord.wait_for(lambda s: s.request_id >= rids[-1])
    assert s.q[0] == pytest.approx(1.0)
    assert len(coord.fake.applied) < 300  # coalesced, not one apply per event
    assert rids == sorted(rids)


def test_errors_are_surfaced_not_swallowed(coord):
    seen = []
    coord.subscribe_errors(seen.append)
    rid = coord.submit_joint("door_hinge", 9.0)
    coord.wait_for(lambda s: s.request_id >= rid)
    assert seen and "outside URDF limits" in seen[-1].message
    assert coord.alive  # owner thread survived


def test_playback_moves_and_manual_input_pauses(coord):
    coord.play("door_hinge")
    coord.wait_for(lambda s: s.q[0] > 0.2, timeout=5)
    rid = coord.submit_joint("door_hinge", 0.1)
    s = coord.wait_for(lambda s: s.request_id >= rid)
    assert not coord.playback.playing and "paused by manual" in s.note
    time.sleep(0.1)
    assert coord.snapshot().q[0] == pytest.approx(0.1)


def test_idle_does_not_busy_loop(coord):
    n = coord.publish_count
    time.sleep(0.3)
    assert coord.publish_count == n


def test_dynamics_mode_routes_slider_to_drive(coord):
    coord.set_mode("dynamics")
    coord.wait_for(lambda s: s.mode == "dynamics")
    rid = coord.submit_joint("door_hinge", 1.0)
    coord.wait_for(lambda s: s.request_id >= rid)
    assert coord.snapshot().q[0] == 0.0                    # no teleport in dynamics mode
    coord.physics_step(5)
    s = coord.wait_for(lambda s: s.sim_time > 0)
    assert 0 < s.q[0] <= 1.0


def test_bounded_command_queue():
    gate = threading.Event()

    class Slow(FakeDriver):
        def reset_to_q0(self):
            gate.wait(5)

    c = SimulationCoordinator(lambda: Slow(), playback_hz=50).start()
    try:
        c.reset()
        time.sleep(0.05)
        with pytest.raises(CommandQueueFull):
            for _ in range(SimulationCoordinator.MAX_COMMANDS + 5):
                c.pause()
    finally:
        gate.set()
        c.stop()


def test_stop_closes_driver_in_owner_thread(coord):
    coord.stop()
    assert coord.fake.closed and not coord.alive
