import omniisaacgymenvs.robots.articulations.utils.MFP_utils as pxr_utils
from pxr import Usd, Gf, UsdShade, UsdPhysics
from scipy.spatial.transform import Rotation
from dataclasses import dataclass, field
from typing import Tuple
import math

class TypeFactoryBuilder:
    def __init__(self):
        self.creators = {}

    def register_instance(self, type):
        self.creators[type.__name__] = type

    def register_instance_by_name(self, name, type):
        self.creators[name] = type

    def get_item(self, params):
        assert "name" in list(params.keys()), "The name of the type must be provided."
        assert params["name"] in self.creators, "Unknown type."
        return self.creators[params["name"]](**params)


####################################################################################################
## Define the types of the geometric primitives
####################################################################################################


@dataclass
class GeometricPrimitive:
    """
    Base class for geometric primitives.
    Geometric primitives are used to define the shape of the robots
    and their composing elements.

    Args:
        name (str): The name of the primitive. This is used to identify the
            type of the primitive. It allows to create the primitive using
            the factory.
        refinement (int): The refinement level of the primitive. A higher
            refinement level results in a smoother shape.
        has_collider (bool): Whether the primitive has a collider. If True,
            the primitive will be used for collision detection.
        is_rigid (bool): Whether the primitive is rigid. If True, the primitive
            will be used for physics simulation.
        marker_scale (float): The scale of the markers. Markers are used to
            visualize the position and orientation of the primitive in the scene.
    """

    refinement: int = 2
    has_collider: bool = False
    is_rigid: bool = False
    marker_scale: float = 1.0

    def __post_init__(self) -> None:
        assert self.refinement > 0, "The refinement level must be larger than 0."
        self.refinement = int(self.refinement)

    def build(self, stage: Usd.Stage, path: str = None) -> Tuple[str, Usd.Prim]:
        """
        Builds the geometric primitive.
        
        Args:
            stage (Usd.Stage): The USD stage.
            path (str): The path to the primitive.
        
        Returns:
            Tuple[str, Usd.Prim]: The path and the prim of the primitive.
        """
        raise NotImplementedError

    def add_positional_marker(
        self, stage: Usd.Stage, path: str, color: UsdShade.Material
    ) -> None:
        """
        Adds a positional marker to the primitive.
        This is used to visualize the position of the primitive in the scene
        and see how well it aligns with its targets.
        
        Args:
            stage (Usd.Stage): The USD stage.
            path (str): The path to the primitive.
            color (UsdShade.Material): The color of the marker.
        """
        raise NotImplementedError

    def add_orientation_marker(
        self, stage: Usd.Stage, path: str, color: UsdShade.Material
    ) -> None:
        """
        Adds an orientation marker to the primitive.
        This is used to visualize the orientation of the primitive in the scene
        and see how well it aligns with its targets.
        
        Args:
            stage (Usd.Stage): The USD stage.
            path (str): The path to the primitive.
            color (UsdShade.Material): The color of the marker
        """
        raise NotImplementedError


@dataclass
class Cylinder(GeometricPrimitive):
    """
    A cylinder geometric primitive.

    Args:
        name (str): The name of the primitive. This is used to identify the
            type of the primitive. It allows to create the primitive using
            the factory.
        refinement (int): The refinement level of the primitive. A higher
            refinement level results in a smoother shape.
        has_collider (bool): Whether the primitive has a collider. If True,
            the primitive will be used for collision detection.
        is_rigid (bool): Whether the primitive is rigid. If True, the primitive
            will be used for physics simulation.
        marker_scale (float): The scale of the markers. Markers are used to
            visualize the position and orientation of the primitive in the scene.
        radius (float): The radius of the cylinder.
        height (float): The height of the cylinder
    """
    name: str = "Cylinder"
    radius: float = 0.1
    height: float = 0.1

    def __post_init__(self) -> None:
        assert self.radius > 0, "The radius must be larger than 0."
        assert self.height > 0, "The height must be larger than 0."
        assert self.refinement > 0, "The refinement level must be larger than 0."

    def build(self, stage: Usd.Stage, path: str = None) -> Tuple[str, Usd.Prim]:
        """
        Builds a cylinder primitive.
        
        Args:
            stage (Usd.Stage): The USD stage.
            path (str): The path to the primitive.
        
        Returns:
            Tuple[str, Usd.Prim]: The path and the prim of the primitive.
        """

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
        """
        Adds a positional marker to the primitive.
        This is used to visualize the position of the primitive in the scene
        and see how well it aligns with its targets.
        
        Args:
            stage (Usd.Stage): The USD stage.
            path (str): The path to the primitive.
            color (UsdShade.Material): The color of the marker.
        """

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
        """
        Adds an orientation marker to the primitive.
        This is used to visualize the orientation of the primitive in the scene
        and see how well it aligns with its targets.
        
        Args:
            stage (Usd.Stage): The USD stage.
            path (str): The path to the primitive.
            color (UsdShade.Material): The color of the marker
        """

        pxr_utils.createArrow(
            stage,
            path,
            0.1 * self.marker_scale,
            0.5 * self.marker_scale,
            [self.radius, 0, 0],
            self.refinement,
        )
        marker_prim = stage.GetPrimAtPath(path)
        pxr_utils.applyMaterial(marker_prim, color)


@dataclass
class Sphere(GeometricPrimitive):
    """
    A sphere geometric primitive.

    Args:
        name (str): The name of the primitive. This is used to identify the
            type of the primitive. It allows to create the primitive using
            the factory.
        refinement (int): The refinement level of the primitive. A higher
            refinement level results in a smoother shape.
        has_collider (bool): Whether the primitive has a collider. If True,
            the primitive will be used for collision detection.
        is_rigid (bool): Whether the primitive is rigid. If True, the primitive
            will be used for physics simulation.
        marker_scale (float): The scale of the markers. Markers are used to
            visualize the position and orientation of the primitive in the scene.
        radius (float): The radius of the sphere.
    """
    name: str = "Sphere"
    radius: float = 0.1

    def __post_init__(self) -> None:
        assert self.radius > 0, "The radius must be larger than 0."
        assert self.refinement > 0, "The refinement level must be larger than 0."
        self.refinement = int(self.refinement)

    def build(self, stage: Usd.Stage, path: str = None) -> Tuple[str, Usd.Prim]:
        """
        Builds a sphere primitive.
        
        Args:
            stage (Usd.Stage): The USD stage.
            path (str): The path to the primitive.
        
        Returns:
            Tuple[str, Usd.Prim]: The path and the prim of the primitive.
        """

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
        """
        Adds a positional marker to the primitive.
        This is used to visualize the position of the primitive in the scene
        and see how well it aligns with its targets.
        
        Args:
            stage (Usd.Stage): The USD stage.
            path (str): The path to the primitive.
            color (UsdShade.Material): The color of the marker.
        """

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
        """
        Adds an orientation marker to the primitive.
        This is used to visualize the orientation of the primitive in the scene
        and see how well it aligns with its targets.
        
        Args:
            stage (Usd.Stage): The USD stage.
            path (str): The path to the primitive.
            color (UsdShade.Material): The color of the marker
        """

        marker_path, marker_prim = pxr_utils.createXform(stage, path)
        pxr_utils.createArrow(
            stage,
            marker_path + "/marker_arrow",
            0.1 * self.marker_scale,
            0.5 * self.marker_scale,
            [self.radius, 0, 0],
            self.refinement,
        )
        pxr_utils.applyMaterial(marker_prim, color)


@dataclass
class Capsule(GeometricPrimitive):
    """
    A capsule geometric primitive.

    Args:
        name (str): The name of the primitive. This is used to identify the
            type of the primitive. It allows to create the primitive using
            the factory.
        refinement (int): The refinement level of the primitive. A higher
            refinement level results in a smoother shape.
        has_collider (bool): Whether the primitive has a collider. If True,
            the primitive will be used for collision detection.
        is_rigid (bool): Whether the primitive is rigid. If True, the primitive
            will be used for physics simulation.
        marker_scale (float): The scale of the markers. Markers are used to
            visualize the position and orientation of the primitive in the scene.
        radius (float): The radius of the capsule.
        height (float): The height of the capsule.
    """

    name: str = "Capsule"
    radius: float = 0.1
    height: float = 0.1

    def __post_init__(self) -> None:
        assert self.radius > 0, "The radius must be larger than 0."
        assert self.height > 0, "The height must be larger than 0."
        self.refinement = int(self.refinement)

    def build(self, stage: Usd.Stage, path: str = None) -> Tuple[str, Usd.Prim]:
        """
        Builds a capsule primitive.
        
        Args:
            stage (Usd.Stage): The USD stage.
            path (str): The path to the primitive.
        
        Returns:
            Tuple[str, Usd.Prim]: The path and the prim of the primitive.
        """

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
        """
        Adds a positional marker to the primitive.
        This is used to visualize the position of the primitive in the scene
        and see how well it aligns with its targets.
        
        Args:
            stage (Usd.Stage): The USD stage.
            path (str): The path to the primitive.
            color (UsdShade.Material): The color of the marker.
        """

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
        """
        Adds an orientation marker to the primitive.
        This is used to visualize the orientation of the primitive in the scene
        and see how well it aligns with its targets.
        
        Args:
            stage (Usd.Stage): The USD stage.
            path (str): The path to the primitive.
            color (UsdShade.Material): The color of the marker
        """

        marker_path, marker_prim = pxr_utils.createXform(stage, path)
        pxr_utils.createArrow(
            stage,
            marker_path + "/marker_arrow",
            0.1 * self.marker_scale,
            0.5 * self.marker_scale,
            [self.radius, 0, 0],
            self.refinement,
        )
        pxr_utils.applyMaterial(marker_prim, color)


@dataclass
class Cube(GeometricPrimitive):
    """
    A cube geometric primitive.

    Args:
        name (str): The name of the primitive. This is used to identify the
            type of the primitive. It allows to create the primitive using
            the factory.
        refinement (int): The refinement level of the primitive. A higher
            refinement level results in a smoother shape.
        has_collider (bool): Whether the primitive has a collider. If True,
            the primitive will be used for collision detection.
        is_rigid (bool): Whether the primitive is rigid. If True, the primitive
            will be used for physics simulation.
        marker_scale (float): The scale of the markers. Markers are used to
            visualize the position and orientation of the primitive in the scene.
        depth (float): The depth of the cube.
        width (float): The width of the cube.
        height (float): The height of the cube.
    """
    name: str = "Cube"
    depth: float = 0.1
    width: float = 0.1
    height: float = 0.1

    def __post_init__(self) -> None:
        assert self.depth > 0, "The depth must be larger than 0."
        assert self.width > 0, "The width must be larger than 0."
        assert self.height > 0, "The height must be larger than 0."
        assert self.refinement > 0, "The refinement level must be larger than 0."
        self.refinement = int(self.refinement)

    def build(self, stage: Usd.Stage, path: str = None) -> Tuple[str, Usd.Prim]:
        """
        Builds a cube primitive.
        
        Args:
            stage (Usd.Stage): The USD stage.
            path (str): The path to the primitive.
        
        Returns:
            Tuple[str, Usd.Prim]: The path and the prim of the primitive.
        """

        path, prim = pxr_utils.createXform(stage, path)
        body_path, body_geom = pxr_utils.createCube(
            stage, path + "/body", self.depth, self.width, self.height, self.refinement
        )
        if self.has_collider:
            prim = stage.GetPrimAtPath(body_path)
            pxr_utils.applyCollider(prim, enable=True)
        if self.is_rigid:
            prim = stage.GetPrimAtPath(path)
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
        """
        Adds a positional marker to the primitive.
        This is used to visualize the position of the primitive in the scene
        and see how well it aligns with its targets.
        
        Args:
            stage (Usd.Stage): The USD stage.
            path (str): The path to the primitive.
            color (UsdShade.Material): The color of the marker.
        """

        pxr_utils.setTranslate(sphere_geom, Gf.Vec3d([0, 0, self.height / 2]))
        pxr_utils.applyMaterial(marker_prim, color)

    def add_orientation_marker(
        self, stage: Usd.Stage, path: str, color: UsdShade.Material
    ) -> None:
        """
        Adds an orientation marker to the primitive.
        This is used to visualize the orientation of the primitive in the scene
        and see how well it aligns with its targets.
        
        Args:
            stage (Usd.Stage): The USD stage.
            path (str): The path to the primitive.
            color (UsdShade.Material): The color of the marker
        """

        marker_path, marker_prim = pxr_utils.createXform(stage, path)
        pxr_utils.createArrow(
            stage,
            marker_path + "/marker_arrow",
            0.1 * self.marker_scale,
            0.5 * self.marker_scale,
            [self.depth / 2, 0, 0],
            self.refinement,
        )
        pxr_utils.applyMaterial(marker_prim, color)


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
    """
    A struct to define a simple color texture.

    Args:
        r (float): The red channel of the color.
        g (float): The green channel of the color.
        b (float): The blue channel of the color.
        roughness (float): The roughness of the material.
    """

    r: float = 0.0
    g: float = 0.0
    b: float = 0.0
    roughness: float = 0.5

    def __post_init__(self) -> None:
        assert 0 <= self.r <= 1, "The red channel must be between 0 and 1."
        assert 0 <= self.g <= 1, "The green channel must be between 0 and 1."
        assert 0 <= self.b <= 1, "The blue channel must be between 0 and 1."
        assert 0 <= self.roughness <= 1, "The roughness must be between 0 and 1."


@dataclass
class PhysicsMaterial:
    """
    A struct to define and create a physics material.
    Note that there currently are issues with patch friction inside
    Omniverse Isaac Sim making it hard to model realistic friction.

    Args:
        static_friction (float): The static friction of the material.
        dynamic_friction (float): The dynamic friction of the material.
        restitution (float): The restitution of the material.
        friction_combine_mode (str): The combine mode for the friction.
        restitution_combine_mode (str): The combine mode for the restitution
    """

    static_friction: float = 0.5
    dynamic_friction: float = 0.5
    restitution: float = 0.5
    friction_combine_mode: str = "average"
    restitution_combine_mode: str = "average"

    def __post_init__(self) -> None:
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

    def build(self, stage, material_path) -> UsdShade.Material:
        """
        Builds the physics material.
        
        Args:
            stage (Usd.Stage): The USD stage.
            material_path (str): The path to the material.
        """
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
    """
    A prismatic joint actuator.

    Args:
        name (str): The name of the actuator. This is used to identify the
            type of the actuator. It allows to create the actuator using
            the factory.
        axis (str): The axis of the actuator. The axis can be "X", "Y", or "Z".
            It defines the direction of the actuator.
        lower_limit (float): The lower limit of the actuator. If None, the actuator
            has no lower limit.
        upper_limit (float): The upper limit of the actuator. If None, the actuator
            has no upper limit.
        velocity_limit (float): The velocity limit of the actuator. If None, the actuator
            has no velocity limit.
        enable_drive (bool): Whether the actuator is enabled. If True, the actuator
            can be driven and controlled by the user.
        force_limit (float): The force limit of the actuator. If None, the actuator
            has no force limit.
        damping (float): The damping of the actuator. The damping is used to dampen
            the movement of the actuator. It is only used if the actuator is driven.
        stiffness (float): The stiffness of the actuator. The stiffness is used to
            stiffen the actuator. It is only used if the actuator is driven.
    """

    name: str = "PrismaticActuator"
    axis: str = "X"
    lower_limit: float = None
    upper_limit: float = None
    velocity_limit: float = None
    enable_drive: bool = False
    force_limit: float = None
    damping: float = 1e10
    stiffness: float = 0.0

    def __post_init__(self) -> None:
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
        """
        Builds the prismatic joint.
        
        Args:
            stage (Usd.Stage): The USD stage.
            joint_path (str): The path to the joint.
            body1_path (str): The path to the first body.
            body2_path (str): The path to the second body.
        """

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
    """
    A revolute joint actuator.

    Args:
        name (str): The name of the actuator. This is used to identify the
            type of the actuator. It allows to create the actuator using
            the factory.
        axis (str): The axis of the actuator. The axis can be "X", "Y", or "Z".
            It defines the direction of the actuator.
        lower_limit (float): The lower limit of the actuator. If None, the actuator
            has no lower limit.
        upper_limit (float): The upper limit of the actuator. If None, the actuator
            has no upper limit.
        velocity_limit (float): The velocity limit of the actuator. If None, the actuator
            has no velocity limit.
        enable_drive (bool): Whether the actuator is enabled. If True, the actuator
            can be driven and controlled by the user.
        force_limit (float): The force limit of the actuator. If None, the actuator
            has no force limit.
        damping (float): The damping of the actuator. The damping is used to dampen
            the movement of the actuator. It is only used if the actuator is driven.
        stiffness (float): The stiffness of the actuator. The stiffness is used to
            stiffen the actuator. It is only used if the actuator is driven.
    """

    name: str = "RevoluteActuator"
    axis: str = "X"
    lower_limit: float = None
    upper_limit: float = None
    velocity_limit: float = None
    enable_drive: bool = False
    force_limit: float = None
    damping: float = 1e10
    stiffness: float = 0.0

    def __post_init__(self) -> None:
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
        """
        Builds the revolute joint.
        
        Args:
            stage (Usd.Stage): The USD stage.
            joint_path (str): The path to the joint.
            body1_path (str): The path to the first body.
            body2_path (str): The path to the second body.
        """

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
## Dynamics, Limits & Actuator definitions
####################################################################################################

@dataclass
class DynamicsCfg:
    """
    Base class for dynamics configurations.

    Args:
        name (str): The name of the dynamics. This is used to identify the
            type of the dynamics. It allows to create the dynamics using
            the factory.
    """
    name: str = "None"


@dataclass
class ZeroOrderDynamicsCfg(DynamicsCfg):
    """
    Zero-order dynamics for the actuators.

    Args:
        name (str): The name of the dynamics. This is used to identify the
            type of the dynamics. It allows to create the dynamics using
            the factory.
    """
    name: str = "zero_order"


@dataclass
class FirstOrderDynamicsCfg(DynamicsCfg):
    """
    First-order dynamics for the actuators.

    Args:
        name (str): The name of the dynamics. This is used to identify the
            type of the dynamics. It allows to create the dynamics using
            the factory.
        time_constant (float): The time constant of the dynamics.
    """
    time_constant: float = 0.2
    name: str = "first_order"

    def __post_init__(self) -> None:
        assert self.time_constant > 0, "Invalid time constant, should be greater than 0"


@dataclass
class SecondOrderDynamicsCfg(DynamicsCfg):
    """
    Second-order dynamics for the actuators.

    Args:
        name (str): The name of the dynamics. This is used to identify the
            type of the dynamics. It allows to create the dynamics using
            the factory.
        natural_frequency (float): The natural frequency of the dynamics.
        damping_ratio (float): The damping ratio of the dynamics.
    """
    natural_frequency: float = 100
    damping_ratio: float = 1 / math.sqrt(2)
    name: str = "second_order"

    def __post_init__(self) -> None:
        assert self.natural_frequency > 0, "Invalid natural frequency, should be greater than 0"
        assert self.damping_ratio > 0, "Invalid damping ratio, should be greater than 0"

DynamicsFactory = TypeFactoryBuilder()
DynamicsFactory.register_instance_by_name("zero_order", ZeroOrderDynamicsCfg)
DynamicsFactory.register_instance_by_name("first_order", FirstOrderDynamicsCfg)
DynamicsFactory.register_instance_by_name("second_order", SecondOrderDynamicsCfg)

@dataclass
class ControlLimitsCfg:
    """
    Control limits for the system.

    Args:
        limits (tuple): The limits of the control. The limits should be a tuple
            of length 2 with the first element being the minimum value and the
            second element being the maximum value.
    """

    limits: tuple = field(default_factory=tuple)

    def __post_init__(self) -> None:
        assert self.limits[0] < self.limits[1], "Invalid limits, min should be less than max"
        assert len(self.limits) == 2, "Invalid limits shape, should be a tuple of length 2"


@dataclass
class ActuatorCfg:
    """
    Actuator configuration.

    Args:
        dynamics (dict): The dynamics of the actuator.
        limits (dict): The limits of the actuator.
    """
    dynamics: dict = field(default_factory=dict)
    limits: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.dynamics = DynamicsFactory.get_item(self.dynamics)
        self.limits = ControlLimitsCfg(**self.limits)

####################################################################################################
## Define the type of high level active joints
####################################################################################################

@dataclass
class Steering:
    """
    A steering joint actuator.
    
    Args:
        limits (tuple): The limits of the steering joint. The limits should be a tuple
            of length 2 with the first element being the minimum value and the
            second element being the maximum value.
        damping (float): The damping of the steering joint.
        stiffness (float): The stiffness of the steering joint.
    """

    limits: tuple = (-30, 30)
    damping: float = 1e10
    stiffness: float = 0

    def __post_init__(self) -> None:
        # Create the revolute joint actuator
        revolute_joint = {
            "name": "RevoluteJoint",
            "axis": "Z",
            "lower_limit": self.limits[0],
            "upper_limit": self.limits[1],
            "velocity_limit": None,
            "enable_drive": True,
            "force_limit": None,
            "damping": self.damping,
            "stiffness": self.stiffness,
        }
        self.actuator = JointActuatorFactory.get_item(revolute_joint)

    def build(
        self,
        stage: Usd.Stage,
        joint_path: str = None,
        path: str = None,
        body_path: str = None,
    ) -> Tuple[str, Usd.Prim]:
        """
        Builds the steering joint. Note that the joint itself is created in a separate
        function to allow for the creation of the steering joint to be called after the
        creation of the other bodies.
        
        Args:
            stage (Usd.Stage): The USD stage.
            joint_path (str): The path to the joint.
            path (str): The path to the steering joint.
            body_path (str): The path to the body.

        Returns:
            Tuple[str, Usd.Prim]: The path and the prim of the steering joint.
        """

        steering_path, steering_prim = pxr_utils.createXform(stage, path)
        self.steering_path = steering_path
        self.body_path = body_path
        self.joint_path = joint_path

        pxr_utils.applyRigidBody(steering_prim)
        pxr_utils.applyMass(steering_prim, 0.05)
        return steering_path, steering_prim

    def create_joints(self, stage: Usd.Stage) -> None:
        """
        Creates the steering joint.

        Args:
            stage (Usd.Stage): The USD stage.
        """

        self.actuator.build(stage, self.joint_path, self.body_path, self.steering_path)


@dataclass
class Suspension:
    """
    A suspension joint. We model the suspension as a prismatic joint with a spring
    and a damping element. Note that the suspension joint is a passive joint and
    should not be driven by the user. Though, to enable the suspension joint to
    act like a spring-damper system, we need to enable the drive of the joint.
    Hence it will show-up as a drivable joint.

    Args:
        travel (float): The travel of the suspension joint.
        damping (float): The damping of the suspension joint.
        stiffness (float): The stiffness of the suspension joint.
    """

    travel: float = 0.1
    damping: float = 1e10
    stiffness: float = 0.0

    def __post_init__(self) -> None:
        assert self.damping >= 0, "The damping must be larger than 0."
        assert self.stiffness >= 0, "The stiffness must be larger than 0."
        # Hard-coded definition of the suspension elements.

        piston_shape = {
            "name": "Cylinder",
            "height": self.travel,
            "radius": self.travel / 6,
            "has_collider": False,
            "is_rigid": False,
            "refinement": 2,
        }
        rod_shape = {
            "name": "Cylinder",
            "height": self.travel,
            "radius": self.travel / 8,
            "has_collider": False,
            "is_rigid": False,
            "refinement": 2,
        }
        pristmatic_joint = {
            "name": "PrismaticJoint",
            "axis": "Z",
            "lower_limit": 0.0,
            "upper_limit": self.travel,
            "velocity_limit": None,
            "enable_drive": True,
            "force_limit": None,
            "damping": self.damping,
            "stiffness": self.stiffness,
        }
        self.piston_shape = Cylinder(**piston_shape)
        self.rod_shape = Cylinder(**rod_shape)
        self.spring = PrismaticJoint(**pristmatic_joint)

    def build(self, stage: Usd.Stage, joint_path: str, path: str, body_path: str, offset: Tuple[float, float, float]) -> Tuple[str, Usd.Prim]:
        """
        Builds the suspension joint. Note that the joint itself is created in a separate
        
        Args:
            stage (Usd.Stage): The USD stage.
            joint_path (str): The path to the joint.
            path (str): The path to the suspension joint.
            body_path (str): The path to the body.
            offset (tuple): The offset of the suspension joint.
        
        Returns:
            Tuple[str, Usd.Prim]: The path and the prim of the suspension joint.
        """    
        # Create Xform to store the suspension
        path, prim = pxr_utils.createXform(stage, path)
        self.body_path = body_path
        self.joint_path = joint_path

        # Build piston sleeve
        self.piston_path, piston_prim = self.piston_shape.build(stage, path+"/piston_sleeve")
        pxr_utils.applyRigidBody(piston_prim)
        pxr_utils.applyMass(piston_prim, 0.001)
        pxr_utils.setTranslate(piston_prim, Gf.Vec3d(offset[0], offset[1], offset[2] + self.travel*3/2))

        # Build piston rod
        self.rod_path, rod_prim = self.rod_shape.build(stage, path + "/rod")
        pxr_utils.applyRigidBody(rod_prim)
        pxr_utils.applyMass(rod_prim, 0.001)
        pxr_utils.setTranslate(rod_prim, Gf.Vec3d(offset[0], offset[1], offset[2] + self.travel/2))
        return self.rod_path, rod_prim

    def create_joints(self, stage: Usd.Stage) -> None:
        """
        Creates the suspension joint.
        
        Args:
            stage (Usd.Stage): The USD stage.
        """

        # Build the joint between the body and the piston
        pxr_utils.createFixedJoint(
            stage, self.joint_path + "_piston", self.body_path, self.piston_path
        )
        # Build the joint to create the spring
        self.spring.build(stage, self.joint_path + "_spring", self.piston_path, self.rod_path)


@dataclass
class Wheel:
    """
    A wheel with a direct drive actuator.
    
    Args:
        visual_shape (dict): The visual shape of the wheel.
        collider_shape (dict): The collider shape of the wheel.
        mass (float): The mass of the wheel.
    """

    visual_shape: GeometricPrimitive = field(default_factory=dict)
    collider_shape: GeometricPrimitive = field(default_factory=dict)
    mass: float = 1.0
    # visual_material: SimpleColorTexture = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Force the collision shape to have a collider
        self.collider_shape["has_collider"] = True
        # Force the visual and collision shapes to be non-rigid
        self.collider_shape["is_rigid"] = False
        self.visual_shape["is_rigid"] = False

        self.visual_shape = GeometricPrimitiveFactory.get_item(self.visual_shape)
        self.collider_shape = GeometricPrimitiveFactory.get_item(self.collider_shape)

    def build(self, stage: Usd.Stage, path: str = None) -> Tuple[str, Usd.Prim, str, Usd.Prim]:
        """
        Builds the wheel.
        
        Args:
            stage (Usd.Stage): The USD stage.
            path (str): The path to the wheel.
        
        Returns:
            Tuple[str, Usd.Prim, str, Usd.Prim]: The path and the prim of the wheel and the collider of that wheel.
        """

        wheel_path, wheel_prim = pxr_utils.createXform(stage, path)
        visual_path, visual_prim = self.visual_shape.build(stage, path + "/visual")
        collider_path, collider_prim = self.collider_shape.build(
            stage, path + "/collision"
        )
        collider_prim.GetAttribute("visibility").Set("invisible")
        pxr_utils.applyRigidBody(wheel_prim)
        pxr_utils.applyMass(wheel_prim, self.mass)
        return wheel_path, wheel_prim, collider_path, collider_prim


@dataclass
class DirectDriveWheel:
    """
    A direct drive wheel with a revolute joint actuator.
    
    Args:
        wheel (dict): The wheel configuration.
        actuator (dict): The actuator configuration.
        offset (tuple): The offset of the wheel.
        orientation (tuple): The orientation of the wheel.
    """

    wheel: Wheel = field(default_factory=dict)
    actuator: RevoluteJoint = field(default_factory=dict)
    offset: Tuple = (0, 0, 0)
    orientation: Tuple = (0, 90, 0)

    def __post_init__(self):
        self.wheel = Wheel(**self.wheel)
        self.actuator = JointActuatorFactory.get_item(self.actuator)
        self.collider = None

    def build(
        self,
        stage: Usd.Stage,
        joint_path: str = None,
        wheel_path: str = None,
        body_path: str = None,
    ) -> Tuple[str, Usd.Prim]:
        """
        Builds the direct drive wheel.

        Args:
            stage (Usd.Stage): The USD stage.
            joint_path (str): The path to the joint.
            wheel_path (str): The path to the wheel.
            body_path (str): The path to the body.
        
        Returns:
            Tuple[str, Usd.Prim]: The path and the prim of the wheel.
        """

        # Create the wheel
        wheel_path, wheel_prim, wheel_collider_path, wheel_collider_prim = self.wheel.build(stage, wheel_path)
        pxr_utils.setTranslate(wheel_prim, Gf.Vec3d(*self.offset))
        q_xyzw = Rotation.from_euler("xyz", self.orientation, degrees=True).as_quat()
        pxr_utils.setOrient(
            wheel_prim, Gf.Quatd(q_xyzw[3], Gf.Vec3d([q_xyzw[0], q_xyzw[1], q_xyzw[2]]))
        )
        # Create the joint
        self.actuator.build(stage, joint_path, body_path, wheel_path)

        return wheel_path, wheel_prim, wheel_collider_path, wheel_collider_prim


@dataclass
class FullyFeaturedWheel:
    """
    A fully featured wheel with a direct drive actuator, a steering joint actuator,
    and a suspension joint.

    Args:
        drive_wheel (dict): The direct drive wheel configuration.
        steering (dict): The steering joint configuration.
        suspension (dict): The suspension joint configuration.
        offset (tuple): The offset of the wheel.
        orientation (tuple): The orientation of the wheel.
    """

    drive_wheel: DirectDriveWheel = field(default_factory=dict)
    steering: Steering = field(default_factory=dict)
    suspension: Suspension = field(default_factory=dict)
    offset: tuple = (0, 0, 0)
    orientation: tuple = (0, 90, 0)

    def __post_init__(self) -> None:
        self.drive_wheel = DirectDriveWheel(**self.drive_wheel)
        if self.suspension is not None:
            self.suspension = Suspension(**self.suspension)
        if self.steering is not None:
            self.steering = Steering(**self.steering)

    def build(self, stage: Usd.Stage, joint_path: str = None, path: str = None, body_path: str = None) -> Tuple[str, Usd.Prim, str, Usd.Prim]:
        """
        Builds the fully featured wheel.
        
        Args:
            stage (Usd.Stage): The USD stage.
            joint_path (str): The path to the joint.
            path (str): The path to the wheel.
            body_path (str): The path to the body.
        
        Returns:
            Tuple[str, Usd.Prim, str, Usd.Prim]: The path and the prim of the wheel and the wheel collider.
        """

        ffw_path, ffw_prim = pxr_utils.createXform(stage, path)
        # Create suspension
        if self.suspension is not None:
            suspension_offset = (0.0,  - math.copysign(self.suspension.travel, self.offset[1]) / 2, 0.0)
            body_path, body_prim = self.suspension.build(stage, joint_path + "_suspension", ffw_path + "/suspension", body_path, suspension_offset)
        # Create steering
        if self.steering is not None:
            body_path, body_prim = self.steering.build(stage, joint_path + "_steering", ffw_path + "/steering", body_path)
        # Create wheel
        wheel_path, wheel_prim, wheel_collider_path, wheel_collider_prim = self.drive_wheel.build(stage, joint_path + "_wheel", ffw_path + "/wheel", body_path)

        # Move the whole wheel to the desired pose
        pxr_utils.setTranslate(ffw_prim, Gf.Vec3d(*self.offset))
        q_xyzw = Rotation.from_euler("xyz", self.orientation, degrees=True).as_quat()
        pxr_utils.setOrient(
            ffw_prim, Gf.Quatd(q_xyzw[3], Gf.Vec3d([q_xyzw[0], q_xyzw[1], q_xyzw[2]]))
        )

        # Create the joints
        if self.suspension is not None:
            self.suspension.create_joints(stage)
        if self.steering is not None:
            self.steering.create_joints(stage)

        return wheel_path, wheel_prim, wheel_collider_path, wheel_collider_prim

####################################################################################################
## Define the type of high level passive joints
####################################################################################################

@dataclass
class ZeroFrictionSphere:
    """
    A passive wheel with zero friction. It is modeled as a sphere
    with a fixed joint actuator. 
    
    Args:
        name (str): The name of the type of passive wheel.
        radius (float): The radius of the sphere.
        mass (float): The mass of the sphere.
        offset (tuple): The offset of the sphere.
    """

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
        """
        Builds the passive wheel.
        
        Args:
            stage (Usd.Stage): The USD stage.
            joint_path (str): The path to the joint.
            material_path (str): The path to the material.
            path (str): The path to the wheel.
            body_path (str): The path to the body.
        
        Returns:
            Tuple[str, Usd.Prim]: The path and the prim of the wheel.
        """

        path, prim = self.shape.build(stage, path)
        pxr_utils.applyMass(prim, self.mass)
        pxr_utils.setTranslate(prim, Gf.Vec3d(*self.offset))
        pxr_utils.createFixedJoint(stage, joint_path, body_path, path)

        if not stage.GetPrimAtPath(material_path).IsValid():
            mat = PhysicsMaterial(**self.zero_friction).build(stage, material_path)
            mat = UsdShade.Material.Get(stage, material_path)
        else:
            mat = UsdShade.Material.Get(stage, material_path)
        pxr_utils.applyMaterial(prim, mat, purpose="physics")
        return None, None, path, prim


@dataclass
class CasterWheel:
    """
    A passive caster wheel with a revolute joint actuator.
    
    Args:
        wheel (dict): The wheel configuration.
        wheel_joint (dict): The wheel joint configuration.
        caster_joint (dict): The caster joint configuration.
        caster_offset (tuple): The offset of the caster wheel.
        wheel_offset (tuple): The offset of the wheel.
        wheel_orientation (tuple): The orientation of the wheel.
    """

    name: str = "CasterWheel"
    wheel: Wheel = field(default_factory=dict)
    wheel_joint: RevoluteJoint = field(default_factory=dict)
    caster_joint: RevoluteJoint = field(default_factory=dict)
    caster_offset: Tuple = (0, 0, 0)
    wheel_offset: Tuple = (0, 0, 0)
    wheel_orientation: Tuple = (90, 0, 0)

    def __post_init__(self) -> None:
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
        material_path: str = None,
        path: str = None,
        body_path: str = None,
    ) -> Tuple[str, Usd.Prim]:
        """
        Builds the caster wheel.
        
        Args:
            stage (Usd.Stage): The USD stage.
            joint_path (str): The path to the joint.
            material_path (str): The path to the material.
            path (str): The path to the caster wheel.
            body_path (str): The path to the body.
        
        Returns:
            Tuple[str, Usd.Prim]: The path and the prim of the caster wheel.
        """

        # Create the xform that will hold the caster wheel
        caster_wheel_path, caster_wheel_prim = pxr_utils.createXform(stage, path)
        # Create the wheel
        wheel_path, wheel_prim, wheel_collider_path, wheel_collider_prim = self.wheel.build(stage, caster_wheel_path + "/wheel")
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
        pxr_utils.applyRigidBody(caster_prim)
        pxr_utils.applyMass(caster_prim, 0.0005)
        pxr_utils.setTranslate(caster_prim, Gf.Vec3d(*self.caster_offset))
        # Create the joints
        self.caster_joint.build(stage, joint_path + "_caster", body_path, caster_path)
        self.wheel_joint.build(stage, joint_path + "_wheel", caster_path, wheel_path)

        return wheel_path, wheel_prim, wheel_collider_path, wheel_collider_prim


PassiveWheelFactory = TypeFactoryBuilder()
PassiveWheelFactory.register_instance(ZeroFrictionSphere)
PassiveWheelFactory.register_instance(CasterWheel)
