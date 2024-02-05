from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.stage import get_current_stage
from pxr import Gf
from omniisaacgymenvs.utils.shape_utils import setScale, setTranslate, setOrient
from omniisaacgymenvs.robots.articulations.utils.MFP_utils import createXform, applyCollider, applyRigidBody

class Dock:
    def __init__(self,  
                 prim_path:str, 
                 usd_path:str, 
                 position:Gf.Vec3d = Gf.Vec3d(0, 0, 0),
                 orientation:Gf.Quatd = Gf.Quatd(1, 0, 0, 0)):
        self.stage = get_current_stage()
        self.prim_path = prim_path
        self.usd_path = usd_path
        self.prim = self.build(position=position, orientation=orientation)
    def build(self, position:Gf.Vec3d, orientation:Gf.Quatd):
        _, prim = createXform(self.stage, self.prim_path)
        setScale(prim, Gf.Vec3d(1, 1, 1))
        setTranslate(prim, position)
        setOrient(prim, orientation)
        add_reference_to_stage(self.usd_path, self.prim_path)
        applyCollider(prim, True)
        applyRigidBody(prim)
        return prim