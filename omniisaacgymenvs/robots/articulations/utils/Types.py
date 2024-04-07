import omniisaacgymenvs.robots.articulations.utils.MFP_utils as pxr_utils
from pxr import Usd, Gf, UsdShade, UsdPhysics
from scipy.spatial.transform import Rotation
from dataclasses import dataclass, field
from typing import Tuple


class TypeFactoryBuilder:
    def __init__(self):
        self.creators = {}

    def register_instance(self, type):
        self.creators[type.__name__] = type

    def get_item(self, params):
        assert "name" in list(params.keys()), "The name of the type must be provided."
        assert params["name"] in self.creators, "Unknown type."
        return self.creators[params["name"]](**params)


####################################################################################################
## Define the types of the geometric primitives
####################################################################################################


@dataclass
class GeometricPrimitive:
    refinement: int = 2
    has_collider: bool = False
    is_rigid: bool = False

    def __post_init__(self):
        assert self.refinement > 0, "The refinement level must be larger than 0."
        self.refinement = int(self.refinement)

    def build(self, stage: Usd.Stage, path: str = None) -> Tuple[str, Usd.Prim]:
        raise NotImplementedError

    def add_positional_marker(
        self, stage: Usd.Stage, path: str, color: UsdShade.Material
    ) -> None:
        raise NotImplementedError

    def add_orientation_marker(
        self, stage: Usd.Stage, path: str, color: UsdShade.Material
    ) -> None:
        raise NotImplementedError


@dataclass
class Cylinder(GeometricPrimitive):
    name: str = "Cylinder"
    radius: float = 0.1
    height: float = 0.1

    def __post_init__(self):
        assert self.radius > 0, "The radius must be larger than 0."
        assert self.height > 0, "The height must be larger than 0."
        assert self.refinement > 0, "The refinement level must be larger than 0."

    def build(self, stage: Usd.Stage, path: str = None) -> Tuple[str, Usd.Prim]:
        path, geom = pxr_utils.createCylinder(
            stage, path, self.radius, self.height, self.refinement
        )
        prim = stage.GetPrimAtPath(path)
        if self.has_collider:
            pxr_utils.applyCollider(prim, enable=True)
        if self.is_rigid:
            pxr_utils.applyRigidBody(prim)
        return path, prim

    def add_positional_marker(
        self, stage: Usd.Stage, path: str, color: UsdShade.Material
    ) -> None:
        marker_path, marker_prim = pxr_utils.createXform(stage, path)
        sphere_path, sphere_geom = pxr_utils.createSphere(
            stage,
            marker_path + "/marker_sphere",
            0.05,
            self.refinement,
        )
        pxr_utils.setTranslate(sphere_geom, Gf.Vec3d([0, 0, self.height / 2]))
        pxr_utils.applyMaterial(marker_prim, color)

    def add_orientation_marker(
        self, stage: Usd.Stage, path: str, color: UsdShade.Material
    ) -> None:
        pxr_utils.createArrow(
            stage,
            path,
            0.1,
            0.5,
            [self.radius, 0, 0],
            self.refinement,
        )
        marker_prim = stage.GetPrimAtPath(path)
        pxr_utils.applyMaterial(marker_prim, color)


@dataclass
class Sphere(GeometricPrimitive):
    name: str = "Sphere"
    radius: float = 0.1

    def __post_init__(self):
        assert self.radius > 0, "The radius must be larger than 0."
        assert self.refinement > 0, "The refinement level must be larger than 0."
        self.refinement = int(self.refinement)

    def build(self, stage: Usd.Stage, path: str = None) -> Tuple[str, Usd.Prim]:
        path, geom = pxr_utils.createSphere(stage, path, self.radius, self.refinement)
        prim = stage.GetPrimAtPath(path)
        if self.has_collider:
            pxr_utils.applyCollider(prim, enable=True)
        if self.is_rigid:
            pxr_utils.applyRigidBody(prim)
        return path, prim

    def add_positional_marker(
        self, stage: Usd.Stage, path: str, color: UsdShade.Material
    ) -> None:
        marker_path, marker_prim = pxr_utils.createXform(stage, path)
        sphere_path, sphere_geom = pxr_utils.createSphere(
            stage,
            marker_path + "/marker_sphere",
            0.05,
            self.refinement,
        )
        pxr_utils.setTranslate(sphere_geom, Gf.Vec3d([0, 0, self.radius]))
        pxr_utils.applyMaterial(marker_prim, color)

    def add_orientation_marker(
        self, stage: Usd.Stage, path: str, color: UsdShade.Material
    ) -> None:
        marker_path, marker_prim = pxr_utils.createXform(stage, path)
        pxr_utils.createArrow(
            stage,
            marker_path + "/marker_arrow",
            0.1,
            0.5,
            [self.radius, 0, 0],
            self.refinement,
        )
        pxr_utils.applyMaterial(marker_prim, color)


@dataclass
class Capsule(GeometricPrimitive):
    name: str = "Capsule"
    radius: float = 0.1
    height: float = 0.1

    def __post_init__(self):
        assert self.radius > 0, "The radius must be larger than 0."
        assert self.height > 0, "The height must be larger than 0."
        self.refinement = int(self.refinement)

    def build(self, stage: Usd.Stage, path: str = None) -> Tuple[str, Usd.Prim]:
        path, geom = pxr_utils.createCapsule(
            stage, path, self.radius, self.height, self.refinement
        )
        prim = stage.GetPrimAtPath(path)
        if self.has_collider:
            pxr_utils.applyCollider(prim, enable=True)
        if self.is_rigid:
            pxr_utils.applyRigidBody(prim)
        return path, prim

    def add_positional_marker(
        self, stage: Usd.Stage, path: str, color: UsdShade.Material
    ) -> None:
        marker_path, marker_prim = pxr_utils.createXform(stage, path)
        sphere_path, sphere_geom = pxr_utils.createSphere(
            stage,
            marker_path + "/marker_sphere",
            0.05,
            self.refinement,
        )
        pxr_utils.setTranslate(
            sphere_geom, Gf.Vec3d([0, 0, self.height / 2 + self.radius])
        )
        pxr_utils.applyMaterial(marker_prim, color)

    def add_orientation_marker(
        self, stage: Usd.Stage, path: str, color: UsdShade.Material
    ) -> None:
        marker_path, marker_prim = pxr_utils.createXform(stage, path)
        pxr_utils.createArrow(
            stage,
            marker_path + "/marker_arrow",
            0.1,
            0.5,
            [self.radius, 0, 0],
            self.refinement,
        )
        pxr_utils.applyMaterial(marker_prim, color)


@dataclass
class Cube(GeometricPrimitive):
    name: str = "Cube"
    width: float = 0.1
    depth: float = 0.1
    height: float = 0.1

    def __post_init__(self):
        assert self.width > 0, "The width must be larger than 0."
        assert self.depth > 0, "The depth must be larger than 0."
        assert self.height > 0, "The height must be larger than 0."
        assert self.refinement > 0, "The refinement level must be larger than 0."
        self.refinement = int(self.refinement)

    def build(self, stage: Usd.Stage, path: str = None) -> Tuple[str, Usd.Prim]:
        path, geom = pxr_utils.createCube(
            stage, path, self.width, self.depth, self.height, self.refinement
        )
        prim = stage.GetPrimAtPath(path)
        if self.has_collider:
            pxr_utils.applyCollider(prim, enable=True)
        if self.is_rigid:
            pxr_utils.applyRigidBody(prim)
        return path, prim

    def add_positional_marker(
        self, stage: Usd.Stage, path: str, color: UsdShade.Material
    ) -> None:
        marker_path, marker_prim = pxr_utils.createXform(stage, path)
        sphere_path, sphere_geom = pxr_utils.createSphere(
            stage,
            marker_path + "/marker_sphere",
            0.05,
            self.refinement,
        )
        pxr_utils.setTranslate(sphere_geom, Gf.Vec3d([0, 0, self.height / 2]))
        pxr_utils.applyMaterial(marker_prim, color)

    def add_orientation_marker(
        self, stage: Usd.Stage, path: str, color: UsdShade.Material
    ) -> None:
        marker_path, marker_prim = pxr_utils.createXform(stage, path)
        pxr_utils.createArrow(
            stage,
            marker_path + "/marker_arrow",
            0.1,
            0.5,
            [self.width / 2, 0, 0],
            self.refinement,
        )
        pxr_utils.applyMaterial(marker_path, color)


GeometricPrimitiveFactory = TypeFactoryBuilder()
GeometricPrimitiveFactory.register_instance(Cylinder)
GeometricPrimitiveFactory.register_instance(Sphere)
GeometricPrimitiveFactory.register_instance(Capsule)
GeometricPrimitiveFactory.register_instance(Cube)

####################################################################################################
## Define the type of physics materials
####################################################################################################


@dataclass
class SimpleColorTexture:
    r: float = 0.0
    g: float = 0.0
    b: float = 0.0
    roughness: float = 0.5

    def __post_init__(self):
        assert 0 <= self.r <= 1, "The red channel must be between 0 and 1."
        assert 0 <= self.g <= 1, "The green channel must be between 0 and 1."
        assert 0 <= self.b <= 1, "The blue channel must be between 0 and 1."
        assert 0 <= self.roughness <= 1, "The roughness must be between 0 and 1."


@dataclass
class PhysicsMaterial:
    static_friction: float = 0.5
    dynamic_friction: float = 0.5
    restitution: float = 0.5
    friction_combine_mode: str = "average"
    restitution_combine_mode: str = "average"

    def __post_init__(self):
        combine_modes = ["average", "min", "max", "multiply"]
        assert (
            0 <= self.static_friction <= 1
        ), "The static friction must be between 0 and 1."
        assert (
            0 <= self.dynamic_friction <= 1
        ), "The dynamic friction must be between 0 and 1."
        assert 0 <= self.restitution <= 1, "The restitution must be between 0 and 1."
        assert (
            self.friction_combine_mode in combine_modes
        ), "The friction combine mode must be one of 'average', 'min', 'max', or 'multiply'."
        assert (
            self.restitution_combine_mode in combine_modes
        ), "The restitution combine mode must be one of 'average', 'min', 'max', or 'multiply'."

    def build(self, stage, material_path):
        material = pxr_utils.createPhysicsMaterial(
            stage,
            material_path,
            static_friction=self.static_friction,
            dynamic_friction=self.dynamic_friction,
            restitution=self.restitution,
            friction_combine_mode=self.friction_combine_mode,
            restitution_combine_mode=self.restitution_combine_mode,
        )
        return material


####################################################################################################
## Define the type of joint actuators
####################################################################################################


@dataclass
class PrismaticJoint:
    name: str = "PrismaticActuator"
    axis: str = "X"
    lower_limit: float = None
    upper_limit: float = None
    velocity_limit: float = None
    enable_drive: bool = False
    force_limit: float = None
    damping: float = 1e10
    stiffness: float = 0.0

    def __post_init__(self):
        if (self.lower_limit is not None) and (self.upper_limit is not None):
            assert (
                self.lower_limit < self.upper_limit
            ), "The lower limit must be smaller than the upper limit."
        if self.velocity_limit is not None:
            assert self.velocity_limit > 0, "The velocity limit must be larger than 0."
        if self.force_limit is not None:
            assert self.force_limit > 0, "The force limit must be larger than 0."
        assert self.damping >= 0, "The damping must be larger than 0."
        assert self.stiffness >= 0, "The stiffness must be larger than or equal to 0."

    def build(
        self,
        stage: Usd.Stage,
        joint_path: str,
        body1_path: str,
        body2_path: str,
    ) -> UsdPhysics.PrismaticJoint:
        joint = pxr_utils.createPrismaticJoint(
            stage,
            joint_path,
            body1_path,
            body2_path,
            axis=self.axis,
            limit_low=self.lower_limit,
            limit_high=self.upper_limit,
            enable_drive=self.enable_drive,
            damping=self.damping,
            stiffness=self.stiffness,
            force_limit=self.force_limit,
        )
        return joint


@dataclass
class RevoluteJoint:
    name: str = "RevoluteActuator"
    axis: str = "X"
    lower_limit: float = None
    upper_limit: float = None
    velocity_limit: float = None
    enable_drive: bool = False
    force_limit: float = None
    damping: float = 1e10
    stiffness: float = 0.0

    def __post_init__(self):
        if (self.lower_limit is not None) and (self.upper_limit is not None):
            assert (
                self.lower_limit < self.upper_limit
            ), "The lower limit must be smaller than the upper limit."
        if self.velocity_limit is not None:
            assert self.velocity_limit > 0, "The velocity limit must be larger than 0."
        if self.force_limit is not None:
            assert self.force_limit > 0, "The force limit must be larger than 0."
        assert self.damping >= 0, "The damping must be larger than 0."
        assert self.stiffness >= 0, "The stiffness must be larger than or equal to 0."

    def build(
        self,
        stage: Usd.Stage,
        joint_path: str,
        body1_path: str,
        body2_path: str,
    ) -> UsdPhysics.RevoluteJoint:
        joint = pxr_utils.createRevoluteJoint(
            stage,
            joint_path,
            body1_path,
            body2_path,
            axis=self.axis,
            limit_low=self.lower_limit,
            limit_high=self.upper_limit,
            enable_drive=self.enable_drive,
            damping=self.damping,
            stiffness=self.stiffness,
            force_limit=self.force_limit,
        )
        return joint


JointActuatorFactory = TypeFactoryBuilder()
JointActuatorFactory.register_instance(PrismaticJoint)
JointActuatorFactory.register_instance(RevoluteJoint)

####################################################################################################
## Define different type of dynamics
####################################################################################################


@dataclass
class ZeroOrderDynamics:
    name: str = "zero_order"


@dataclass
class FirstOrderDynamics:
    name: str = "first_order"
    time_constant: float = 0.1
    delay: float = 0.0

    def __post_init__(self):
        assert self.time_constant > 0, "The time constant must be larger than 0."
        assert self.delay >= 0, "The delay must be larger than or equal to 0."


@dataclass
class SecondOrderDynamics:
    name: str = "second_order"
    damping_ratio: float = 0.7
    natural_frequency: float = 1.0
    delay: float = 0.0

    def __post_init__(self):
        assert (
            0 <= self.damping_ratio <= 1
        ), "The damping ratio must be between 0 and 1."
        assert (
            self.natural_frequency > 0
        ), "The natural frequency must be larger than 0."
        assert self.delay >= 0, "The delay must be larger than or equal to 0."


DynamicsFactory = TypeFactoryBuilder()
DynamicsFactory.register_instance(ZeroOrderDynamics)
DynamicsFactory.register_instance(FirstOrderDynamics)
DynamicsFactory.register_instance(SecondOrderDynamics)

####################################################################################################
## Define the type of high level actuators
####################################################################################################


@dataclass
class Wheel:
    visual_shape: GeometricPrimitive = field(default_factory=dict)
    collider_shape: GeometricPrimitive = field(default_factory=dict)
    mass: float = 1.0
    # physics_material: PhysicsMaterial = field(default_factory=dict)
    # visual_material: SimpleColorTexture = field(default_factory=dict)

    def __post_init__(self):
        # Force the collision shape to have a collider
        self.collider_shape["has_collider"] = True
        # Force the visual and collision shapes to be non-rigid
        self.collider_shape["is_rigid"] = False
        self.visual_shape["is_rigid"] = False

        self.visual_shape = GeometricPrimitiveFactory.get_item(self.visual_shape)
        self.collider_shape = GeometricPrimitiveFactory.get_item(self.collider_shape)
        # self.physics_material = PhysicsMaterial(**self.physics_material)
        # self.visual_material = SimpleColorTexture(**self.visual_material)

    def build(self, stage: Usd.Stage, path: str = None) -> Tuple[str, Usd.Prim]:
        wheel_path, wheel_prim = pxr_utils.createXform(stage, path)
        visual_path, visual_prim = self.visual_shape.build(stage, path + "/visual")
        collider_path, collider_prim = self.collider_shape.build(
            stage, path + "/collision"
        )
        pxr_utils.applyRigidBody(wheel_prim)
        pxr_utils.applyMass(wheel_prim, self.mass)
        # pxr_utils.applyMaterial(visual_prim, self.visual_material)
        # pxr_utils.applyMaterial(collision_prim, self.visual_material)
        return wheel_path, wheel_prim


@dataclass
class DirectDriveWheel:
    wheel: Wheel = field(default_factory=dict)
    actuator: RevoluteJoint = field(default_factory=dict)
    # dynamics: dict = field(default_factory=dict)
    offset: Tuple = (0, 0, 0)
    orientation: Tuple = (0, 90, 0)

    def __post_init__(self):
        self.wheel = Wheel(**self.wheel)
        self.actuator = JointActuatorFactory.get_item(self.actuator)
        # self.dynamics = DynamicsFactory.get_item(self.dynamics)

    def build(
        self,
        stage: Usd.Stage,
        joint_path: str = None,
        wheel_path: str = None,
        body_path: str = None,
    ) -> Tuple[str, Usd.Prim]:
        # Create the wheel
        wheel_path, wheel_prim = self.wheel.build(stage, wheel_path)
        pxr_utils.setTranslate(wheel_prim, Gf.Vec3d(*self.offset))
        q_xyzw = Rotation.from_euler("xyz", self.orientation, degrees=True).as_quat()
        pxr_utils.setOrient(
            wheel_prim, Gf.Quatd(q_xyzw[3], Gf.Vec3d([q_xyzw[0], q_xyzw[1], q_xyzw[2]]))
        )
        # Create the joint
        self.actuator.build(stage, joint_path, body_path, wheel_path)

        return wheel_path, wheel_prim


@dataclass
class ZeroFrictionSphere:
    name: str = "ZeroFrictionSphere"
    radius: float = 0.1
    mass: float = 1.0
    offset: Tuple = (0, 0, 0)

    def __post_init__(self):
        assert self.radius > 0, "The radius must be larger than 0."
        assert self.mass > 0, "The mass must be larger than 0."

        self.zero_friction = {
            "static_friction": 0.0,
            "dynamic_friction": 0.0,
            "restitution": 0.8,
            "friction_combine_mode": "min",
            "restitution_combine_mode": "average",
        }

        shape = {
            "name": "Sphere",
            "radius": self.radius,
            "has_collider": True,
            "is_rigid": True,
            "refinement": 2,
        }
        self.shape = GeometricPrimitiveFactory.get_item(shape)

    def build(
        self,
        stage: Usd.Stage,
        joint_path: str = None,
        material_path: str = None,
        path: str = None,
        body_path: str = None,
    ) -> Tuple[str, Usd.Prim]:
        path, prim = self.shape.build(stage, path)
        pxr_utils.applyMass(prim, self.mass)
        pxr_utils.setTranslate(prim, Gf.Vec3d(*self.offset))
        pxr_utils.createFixedJoint(stage, joint_path, body_path, path)

        if not stage.GetPrimAtPath(material_path).IsValid():
            mat = PhysicsMaterial(**self.zero_friction).build(stage, material_path)
            mat = UsdShade.Material.Get(stage, material_path)
        else:
            mat = UsdShade.Material.Get(stage, material_path)
        print(prim, mat)
        pxr_utils.applyMaterial(prim, mat, purpose="physics")
        return path, prim


@dataclass
class CasterWheel:
    name: str = "CasterWheel"
    wheel: Wheel = field(default_factory=dict)
    wheel_joint: RevoluteJoint = field(default_factory=dict)
    caster_joint: RevoluteJoint = field(default_factory=dict)
    caster_offset: Tuple = (0, 0, 0)
    wheel_offset: Tuple = (0, 0, 0)
    wheel_orientation: Tuple = (90, 0, 0)

    def __post_init__(self):
        self.wheel = Wheel(**self.wheel)
        self.caster_joint["name"] = "RevoluteJoint"
        self.wheel_joint["name"] = "RevoluteJoint"
        self.caster_joint["enable_drive"] = False
        self.wheel_joint["enable_drive"] = False
        self.caster_joint = JointActuatorFactory.get_item(self.caster_joint)
        self.wheel_joint = JointActuatorFactory.get_item(self.wheel_joint)

    def build(
        self,
        stage: Usd.Stage,
        joint_path: str = None,
        path: str = None,
        body_path: str = None,
    ) -> Tuple[str, Usd.Prim]:
        # Create the xform that will hold the caster wheel
        caster_wheel_path, caster_wheel_prim = pxr_utils.createXform(stage, path)
        # Create the wheel
        wheel_path, wheel_prim = self.wheel.build(stage, caster_wheel_path + "/wheel")
        pxr_utils.setTranslate(wheel_prim, Gf.Vec3d(*self.wheel_offset))
        q_xyzw = Rotation.from_euler(
            "xyz", self.wheel_orientation, degrees=True
        ).as_quat()
        pxr_utils.setOrient(
            wheel_prim, Gf.Quatd(q_xyzw[3], Gf.Vec3d([q_xyzw[0], q_xyzw[1], q_xyzw[2]]))
        )
        # Create the caster
        caster_path, caster_prim = pxr_utils.createXform(
            stage, caster_wheel_path + "/caster"
        )
        pxr_utils.setTranslate(caster_prim, Gf.Vec3d(*self.caster_offset))
        # Create the joints
        self.caster_joint.build(stage, joint_path, body_path, caster_path)
        self.wheel_joint.build(stage, joint_path, caster_path, wheel_path)

        return wheel_path, wheel_prim


PassiveWheelFactory = TypeFactoryBuilder()
PassiveWheelFactory.register_instance(ZeroFrictionSphere)
PassiveWheelFactory.register_instance(CasterWheel)
