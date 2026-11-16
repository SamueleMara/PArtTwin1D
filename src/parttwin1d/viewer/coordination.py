"""Thread-safe simulation ownership and snapshot publication for the GUI.

One authoritative simulation state, owned by ONE thread:

1. GUI events (slider, play, step, ...) call ``submit_*``; each gets a
   monotonically increasing request id. Joint targets are *coalesced* (latest
   value per joint wins) and discrete commands go to a bounded queue — rapid
   slider drags cannot build an unbounded backlog.
2. The owner thread applies pending work through ManiSkill (kinematic set or
   PhysX stepping), captures an immutable :class:`StateSnapshot` and publishes
   it (atomic reference swap).
3. Subscribers (viewer re-render, overlays, GUI readouts) are called from the
   owner thread after each publish. Renderers read ``snapshot()`` — they never
   touch the articulation.

Idle with no playback/physics: the thread blocks on a condition variable (no
busy loop). Errors are captured, kept in a bounded log and pushed to error
subscribers — never silently swallowed.

Policy: manual joint input while a kinematic sweep is playing pauses the sweep
(manual input wins; reported in the snapshot note). In dynamics mode with
physics running, joint sliders set the PD drive target instead of teleporting.
"""
from __future__ import annotations

import collections
import threading
import time
import traceback
from dataclasses import dataclass, field


class CommandQueueFull(RuntimeError):
    pass


@dataclass
class PlaybackState:
    playing: bool = False
    loop: bool = True
    speed: float = 1.0
    period_s: float = 4.0       # full open->close cycle at speed 1
    joint: str | None = None
    phase: float = 0.0          # [0,1): triangle wave lower->upper->lower


@dataclass
class DynamicsState:
    running: bool = False
    input_kind: str = "drive"   # "drive" (PD target) | "force" (generalized force)
    drive_target: dict = field(default_factory=dict)
    joint_force: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ErrorEvent:
    time: float
    source: str
    message: str
    detail: str = ""


class SimulationCoordinator:
    MAX_COMMANDS = 128

    def __init__(self, driver_factory, *, playback_hz: float = 30.0, history: int = 20000):
        self._factory = driver_factory
        self.tick_s = 1.0 / float(playback_hz)
        self._cond = threading.Condition()
        self._commands: collections.deque = collections.deque()
        self._joint_targets: dict = {}
        self._req = 0
        self._applied_req = -1
        self._stop = False
        self._snapshot = None
        self._subs: list = []
        self._err_subs: list = []
        self.errors: collections.deque = collections.deque(maxlen=100)
        self.history: collections.deque = collections.deque(maxlen=history)
        self.mode = "kinematic"
        self.playback = PlaybackState()
        self.dynamics = DynamicsState()
        self.driver = None
        self.ready = threading.Event()
        self.failed: ErrorEvent | None = None
        self.publish_count = 0
        self._thread = threading.Thread(target=self._run, name="parttwin-sim-owner", daemon=True)

    # ------------------------------------------------------------------ public
    def start(self, timeout: float = 120.0) -> "SimulationCoordinator":
        self._thread.start()
        if not self.ready.wait(timeout):
            raise TimeoutError("simulation owner did not become ready")
        if self.failed is not None:
            raise RuntimeError(f"simulation start failed: {self.failed.message}\n{self.failed.detail}")
        return self

    def stop(self, timeout: float = 30.0) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify_all()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout)

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    def snapshot(self):
        return self._snapshot  # atomic reference read; snapshot is immutable

    def subscribe(self, fn) -> None:
        self._subs.append(fn)

    def subscribe_errors(self, fn) -> None:
        self._err_subs.append(fn)

    def joints(self):
        return self.driver.joints if self.driver else []

    def wait_for(self, predicate, timeout: float = 10.0):
        """Block until predicate(snapshot) is true (tests / scripted use)."""
        t_end = time.time() + timeout
        while time.time() < t_end:
            s = self._snapshot
            if s is not None and predicate(s):
                return s
            time.sleep(0.01)
        raise TimeoutError("condition on published snapshot not met")

    # ---------------------------------------------------------------- commands
    def _next_req(self) -> int:
        self._req += 1
        return self._req

    def submit_joint(self, name: str, value: float) -> int:
        with self._cond:
            rid = self._next_req()
            self._joint_targets[name] = (float(value), rid)
            self._cond.notify_all()
            return rid

    def submit(self, kind: str, *args) -> int:
        with self._cond:
            if len(self._commands) >= self.MAX_COMMANDS:
                raise CommandQueueFull(f"command queue full ({self.MAX_COMMANDS}); simulation owner is behind")
            rid = self._next_req()
            self._commands.append((kind, args, rid))
            self._cond.notify_all()
            return rid

    # convenience wrappers
    def reset(self):
        return self.submit("reset")

    def play(self, joint: str | None = None):
        return self.submit("play", joint)

    def pause(self):
        return self.submit("pause")

    def step_playback(self, frames: int = 1):
        return self.submit("step", int(frames))

    def set_loop(self, loop: bool):
        return self.submit("loop", bool(loop))

    def set_speed(self, speed: float):
        return self.submit("speed", float(speed))

    def set_mode(self, mode: str):
        return self.submit("mode", mode)

    def set_physics_running(self, running: bool):
        return self.submit("physics_run", bool(running))

    def physics_step(self, n: int = 1):
        return self.submit("physics_step", int(n))

    def set_dynamics_input(self, kind: str, joint: str, value: float):
        return self.submit("dyn_input", kind, joint, float(value))

    # ------------------------------------------------------------ owner thread
    def _report(self, source: str, exc: BaseException | str) -> None:
        if isinstance(exc, BaseException):
            ev = ErrorEvent(time.time(), source, f"{type(exc).__name__}: {exc}", traceback.format_exc())
        else:
            ev = ErrorEvent(time.time(), source, str(exc))
        self.errors.append(ev)
        for fn in list(self._err_subs):
            try:
                fn(ev)
            except Exception:  # pragma: no cover - never let reporting kill the owner
                traceback.print_exc()

    def _publish(self, note: str = "") -> None:
        d = self.driver
        snap = d.snapshot(self.mode, note=note, request_id=self._applied_req)
        self._snapshot = snap
        self.publish_count += 1
        self.history.append((snap.sim_time, snap.wall_time, snap.revision, snap.mode, snap.q.tolist()))
        for fn in list(self._subs):
            try:
                fn(snap)
            except Exception as exc:
                self._report("subscriber", exc)

    def _sweep_q(self, j) -> float:
        ph = self.playback.phase
        tri = 2 * ph if ph < 0.5 else 2 * (1 - ph)
        return j.lower + (j.upper - j.lower) * tri

    def _sweep_joint(self):
        name = self.playback.joint or (self.driver.joints[0].name if self.driver.joints else None)
        for j in self.driver.joints:
            if j.name == name:
                if not j.bounded:
                    raise ValueError(f"cannot sweep unbounded joint {name}")
                return j
        raise ValueError(f"unknown sweep joint {name!r}")

    def _advance_playback(self, frames: float) -> str:
        pb = self.playback
        j = self._sweep_joint()
        pb.phase += frames * self.tick_s * pb.speed / pb.period_s
        if pb.phase >= 1.0:
            if pb.loop:
                pb.phase %= 1.0
            else:
                pb.phase = 0.0
                pb.playing = False
        self.driver.set_joint_kinematic(j.name, self._sweep_q(j))
        return f"sweep {j.name}"

    def _dyn_step(self, n: int) -> None:
        d = self.driver
        dy = self.dynamics
        if dy.input_kind == "drive":
            q = d.get_q()
            for name, v in dy.drive_target.items():
                q[d.joint_index[name]] = v
            d.step_dynamics(n, drive_target=q)
        else:
            import numpy as np

            f = np.zeros(d.dof)
            for name, v in dy.joint_force.items():
                f[d.joint_index[name]] = v
            d.step_dynamics(n, joint_force=f)

    def _apply(self, kind, args, rid) -> str:
        d = self.driver
        pb, dy = self.playback, self.dynamics
        note = ""
        if kind == "reset":
            pb.playing = False
            dy.running = False
            dy.drive_target.clear()
            dy.joint_force.clear()
            pb.phase = 0.0
            d.reset_to_q0()
            note = "reset to q0"
        elif kind == "play":
            if self.mode != "kinematic":
                raise ValueError("kinematic sweep playback requires kinematic mode")
            if args and args[0]:
                pb.joint = args[0]
            j = self._sweep_joint()
            q = d.get_q()[j.index]
            # start the sweep from the current q (on the rising edge)
            pb.phase = 0.5 * (q - j.lower) / (j.upper - j.lower) if j.upper > j.lower else 0.0
            pb.playing = True
            note = f"playing sweep of {j.name}"
        elif kind == "pause":
            pb.playing = False
            note = "paused"
        elif kind == "step":
            if self.mode == "kinematic":
                pb.playing = False
                note = self._advance_playback(args[0])
            else:
                self._dyn_step(args[0])
                note = "physics step"
        elif kind == "loop":
            pb.loop = args[0]
        elif kind == "speed":
            if not 0.01 <= args[0] <= 20:
                raise ValueError("speed must be in [0.01, 20]")
            pb.speed = args[0]
        elif kind == "mode":
            if args[0] not in ("kinematic", "dynamics"):
                raise ValueError(f"unknown mode {args[0]}")
            pb.playing = False
            dy.running = False
            self.mode = args[0]
            if self.mode == "dynamics":
                q = d.get_q()
                dy.drive_target = {j.name: float(q[j.index]) for j in d.joints}
            note = f"mode -> {self.mode}"
        elif kind == "physics_run":
            if self.mode != "dynamics" and args[0]:
                raise ValueError("switch to dynamics mode before running physics")
            dy.running = args[0]
            note = "physics running" if dy.running else "physics paused"
        elif kind == "physics_step":
            if self.mode != "dynamics":
                raise ValueError("physics stepping requires dynamics mode")
            self._dyn_step(args[0])
            note = f"physics +{args[0]} control steps"
        elif kind == "dyn_input":
            kind_in, joint, value = args
            if kind_in not in ("drive", "force"):
                raise ValueError("dynamics input must be 'drive' or 'force'")
            if joint not in d.joint_index:
                raise ValueError(f"unknown joint {joint}")
            dy.input_kind = kind_in
            (dy.drive_target if kind_in == "drive" else dy.joint_force)[joint] = value
            note = f"dynamics input {kind_in} {joint}={value:g}"
        else:
            raise ValueError(f"unknown command {kind!r}")
        return note

    def _run(self) -> None:
        try:
            self.driver = self._factory()
            self._publish("initial state")
        except BaseException as exc:
            self.failed = ErrorEvent(time.time(), "startup", f"{type(exc).__name__}: {exc}", traceback.format_exc())
            self.errors.append(self.failed)
            self.ready.set()
            return
        self.ready.set()
        next_tick = time.perf_counter()
        try:
            while True:
                with self._cond:
                    animating = self.playback.playing or self.dynamics.running
                    if not (self._stop or self._commands or self._joint_targets):
                        if animating:
                            self._cond.wait(max(0.0, next_tick - time.perf_counter()))
                        else:
                            self._cond.wait()
                    if self._stop:
                        break
                    commands = list(self._commands)
                    self._commands.clear()
                    targets = dict(self._joint_targets)
                    self._joint_targets.clear()
                notes = []
                changed = False
                for kind, args, rid in commands:
                    try:
                        notes.append(self._apply(kind, args, rid))
                    except Exception as exc:
                        self._report(f"command:{kind}", exc)
                    self._applied_req = max(self._applied_req, rid)
                    changed = True
                if targets:
                    try:
                        if self.playback.playing:
                            self.playback.playing = False
                            notes.append("sweep paused by manual joint input")
                        if self.mode == "dynamics":
                            for name, (v, _) in targets.items():
                                self.dynamics.drive_target[name] = v
                            self.dynamics.input_kind = "drive"
                            if not self.dynamics.running:
                                notes.append("drive target set (physics paused)")
                        else:
                            q = self.driver.get_q()
                            for name, (v, _) in targets.items():
                                if name not in self.driver.joint_index:
                                    raise KeyError(f"unknown joint {name!r}")
                                q[self.driver.joint_index[name]] = v
                            self.driver.set_q_kinematic(q)
                    except Exception as exc:
                        self._report("joint input", exc)
                    self._applied_req = max(self._applied_req, max(r for _, r in targets.values()))
                    changed = True
                now = time.perf_counter()
                if (self.playback.playing or self.dynamics.running) and now >= next_tick:
                    try:
                        if self.playback.playing:
                            notes.append(self._advance_playback(1))
                        if self.dynamics.running:
                            steps = max(1, round(self.tick_s / self.driver.env.control_timestep))
                            self._dyn_step(steps)
                            notes.append("physics running")
                    except Exception as exc:
                        self.playback.playing = False
                        self.dynamics.running = False
                        self._report("animation", exc)
                    next_tick = now + self.tick_s
                    changed = True
                elif not (self.playback.playing or self.dynamics.running):
                    next_tick = now
                if changed:
                    try:
                        self._publish("; ".join(n for n in notes if n))
                    except Exception as exc:
                        self._report("publish", exc)
        finally:
            try:
                self.driver.close()
            except Exception as exc:
                self._report("shutdown", exc)
