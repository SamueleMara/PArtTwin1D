"""Protect against ManiSkill BaseEnv.get_state_dict assuming an agent exists.

This test models the relevant ManiSkill 3.0.1 call (BaseEnv.__init__ calls
get_state_dict after reset); the full SAPIEN runtime requires a user GPU setup.
"""
import sys
import types
from pathlib import Path


def test_robot_free_init_does_not_touch_absent_controller(monkeypatch):
    import torch

    class FakeScene:
        def get_sim_state(self):
            return {"actors": {}, "articulations": {"door": torch.tensor([[0.]])}}

    class FakeBaseEnv:
        def __init__(self, robot_uids=None, **kwargs):
            assert robot_uids == 'none'
            self.agent = None
            self.scene = FakeScene()
            # ManiSkill invokes get_state_dict at the end of its constructor.
            self.initial_state = self.get_state_dict()
            assert self._get_obs_agent() == {}

        def get_state_dict(self):
            # ManiSkill 3.0.1 default implementation would raise here.
            return self.agent.get_controller_state()

    class FakeSimConfig:
        def __init__(self, **kwargs):
            pass

    m = types.ModuleType('mani_skill')
    m.__path__ = []
    envs = types.ModuleType('mani_skill.envs')
    envs.__path__ = []
    sapien_env = types.ModuleType('mani_skill.envs.sapien_env')
    sapien_env.BaseEnv = FakeBaseEnv
    utils = types.ModuleType('mani_skill.utils')
    utils.__path__ = []
    structs = types.ModuleType('mani_skill.utils.structs')
    structs.__path__ = []
    typesmod = types.ModuleType('mani_skill.utils.structs.types')
    typesmod.SimConfig = FakeSimConfig
    for name, mod in [
        ('mani_skill', m), ('mani_skill.envs', envs),
        ('mani_skill.envs.sapien_env', sapien_env), ('mani_skill.utils', utils),
        ('mani_skill.utils.structs', structs), ('mani_skill.utils.structs.types', typesmod),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)

    from parttwin_smoke.env import make_env_class
    E = make_env_class(Path('/tmp/any.urdf'))
    e = E()
    assert 'articulations' in e.initial_state
    assert e.agent is None


def test_urdf_articulation_builder_gets_explicit_initial_pose(monkeypatch):
    import torch

    class FakePose:
        pass

    sapien = types.ModuleType('sapien')
    sapien.Pose = FakePose
    monkeypatch.setitem(sys.modules, 'sapien', sapien)

    class FakeBuilder:
        initial_pose = None
        def build(self, name):
            assert name == 'parttwin_object'
            assert isinstance(self.initial_pose, FakePose)
            return object()

    class FakeLoader:
        fix_root_link = False
        scale = None
        name = None
        def parse(self, path):
            assert path == '/tmp/foo.urdf'
            assert self.fix_root_link is True
            assert self.name == 'parttwin_object'
            return {'articulation_builders': [FakeBuilder()], 'actor_builders': []}

    class FakeScene:
        def create_urdf_loader(self):
            return FakeLoader()

    from parttwin_smoke.env import make_env_class
    # In this case use a tiny stub module hierarchy again, to isolate the loader.
    m = types.ModuleType('mani_skill'); m.__path__ = []
    envs = types.ModuleType('mani_skill.envs'); envs.__path__ = []
    sapien_env = types.ModuleType('mani_skill.envs.sapien_env')
    sapien_env.BaseEnv = object
    utils = types.ModuleType('mani_skill.utils'); utils.__path__ = []
    structs = types.ModuleType('mani_skill.utils.structs'); structs.__path__ = []
    typesmod = types.ModuleType('mani_skill.utils.structs.types')
    typesmod.SimConfig = object
    for name, mod in [('mani_skill',m), ('mani_skill.envs',envs),
                      ('mani_skill.envs.sapien_env',sapien_env), ('mani_skill.utils',utils),
                      ('mani_skill.utils.structs',structs), ('mani_skill.utils.structs.types',typesmod)]:
        monkeypatch.setitem(sys.modules, name, mod)
    E = make_env_class(Path('/tmp/foo.urdf'))
    e = E.__new__(E)
    e.scene = FakeScene()
    e._load_scene({})
    assert e.articulated_object is not None
