"""Regression for ManiSkill 3.0.1 BaseEnv's unconditional agent serialization."""
import sys
import types
import numpy as np
import torch
import pytest
from parttwin1d.sim.env import build_env_type, set_joint_qpos


class FakeArt:
    def __init__(self):
        self.qpos=torch.zeros((1,1))
        self.qlimits=torch.tensor([[[0.,1.57]]])
        self.links_map={"base_link":object(),"door_link":object()}
    def set_qpos(self,q): self.qpos=q.clone()
    def get_qpos(self): return self.qpos
    def set_qvel(self,q): self.qvel=q.clone()
    def get_active_joints(self): return [types.SimpleNamespace(name='door_hinge')]

class FakeBuilder:
    def __init__(self):
        self.link_builders=[types.SimpleNamespace(visual_records=["unused mesh"])]
    def build(self, **kwargs):
        assert kwargs['name']=='parttwin_object'
        assert self.initial_pose is not None
        assert self.link_builders[0].visual_records == []
        return FakeArt()

class FakeScene:
    def create_urdf_loader(self):
        s=self
        class Loader:
            def parse(self,path): return {"articulation_builders": [FakeBuilder()],"actor_builders": []}
        return Loader()
    def get_sim_state(self): return {"articulations": {"test": torch.zeros((1,3))}}

class FakeBaseEnv:
    def __init__(self,**kwargs):
        assert kwargs['robot_uids']=='none'
        self.agent=None
        self.scene=FakeScene()
        self.num_envs=1
        self.device=torch.device('cpu')
        self.gpu_sim_enabled=False
        self._load_scene({})
        # This reproduces the failing call in BaseEnv.__init__.
        self._init_raw_state=self.get_state_dict()
        self._initialize_episode(torch.tensor([0]),{})


def test_robot_free_scene_serialization(monkeypatch):
    mods={
        'sapien': types.ModuleType('sapien'),
        'mani_skill': types.ModuleType('mani_skill'),
        'mani_skill.envs': types.ModuleType('mani_skill.envs'),
        'mani_skill.envs.sapien_env': types.ModuleType('mani_skill.envs.sapien_env'),
        'mani_skill.utils': types.ModuleType('mani_skill.utils'),
        'mani_skill.utils.structs': types.ModuleType('mani_skill.utils.structs'),
        'mani_skill.utils.structs.types': types.ModuleType('mani_skill.utils.structs.types'),
    }
    mods['sapien'].Pose=lambda **kw:types.SimpleNamespace(**kw)
    mods['mani_skill.envs.sapien_env'].BaseEnv=FakeBaseEnv
    mods['mani_skill.utils.structs.types'].SimConfig=lambda **kw:kw
    for k,v in mods.items():monkeypatch.setitem(sys.modules,k,v)
    cls=build_env_type('fake.urdf',fixed_base=True,T_world_root=np.eye(4))
    env=cls()
    assert 'articulations' in env._init_raw_state
    vals=set_joint_qpos(env,'door_hinge',1.0)
    assert vals=={'door_hinge':1.0}
    assert env.twin.qpos.item()==1
    with pytest.raises(ValueError,match='outside URDF limits'):
        set_joint_qpos(env,'door_hinge',2.0)
