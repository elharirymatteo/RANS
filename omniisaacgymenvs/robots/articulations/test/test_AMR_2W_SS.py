if __name__ == "__main__":
    from omni.isaac.kit import SimulationApp

    cfg = {
        "headless": False,
    }

    simulation_app = SimulationApp(cfg)
    from omni.isaac.core import World
    import omni

    from omniisaacgymenvs.robots.articulations.AMR_2WheelsSkidSteer import (
        AMR_2W_SS,
        SkidSteerParameters,
    )
    from pxr import UsdLux

    timeline = omni.timeline.get_timeline_interface()

    world = World(stage_units_in_meters=1.0)

    world.scene.add_default_ground_plane()
    light = UsdLux.DistantLight.Define(world.stage, "/DistantLight")
    light.CreateIntensityAttr(3000.0)

    physics_ctx = world.get_physics_context()
    physics_ctx.set_solver_type("PGS")

    # Kobuki's Turtlebot 2
    Turtlebot2 = {
        "shape": {
            "name": "Cylinder",
            "radius": 0.354 / 2,
            "height": 0.420,
            "has_collider": True,
            "is_rigid": True,
            "refinement": 2,
        },
        "mass": 6.5,
        "left_wheel": {
            "wheel": {
                "visual_shape": {
                    "name": "Cylinder",
                    "radius": 0.076 / 2,
                    "height": 0.04,
                    "has_collider": False,
                    "is_rigid": False,
                    "refinement": 2,
                },
                "collider_shape": {
                    "name": "Capsule",
                    "radius": 0.076 / 2,
                    "height": 0.04,
                    "has_collider": True,
                    "is_rigid": True,
                    "refinement": 2,
                },
                "mass": 0.05,
            },
            "actuator": {
                "name": "RevoluteJoint",
                "axis": "Z",
                "enable_drive": True,
                "damping": 1e10,
                "stiffness": 0.0,
            },
            "offset": [0.0, -0.24 / 2, -0.420 / 2 + 0.076 / 2 - 0.015],
            "orientation": [-90, 0, 0],
        },
        "right_wheel": {
            "wheel": {
                "visual_shape": {
                    "name": "Cylinder",
                    "radius": 0.076 / 2,
                    "height": 0.04,
                    "has_collider": False,
                    "is_rigid": False,
                    "refinement": 2,
                },
                "collider_shape": {
                    "name": "Capsule",
                    "radius": 0.076 / 2,
                    "height": 0.04,
                    "has_collider": True,
                    "is_rigid": True,
                    "refinement": 2,
                },
                "mass": 0.05,
            },
            "actuator": {
                "name": "RevoluteJoint",
                "axis": "Z",
                "enable_drive": True,
                "damping": 1e10,
                "stiffness": 0.0,
            },
            "offset": [0.0, 0.24 / 2, -0.420 / 2 + 0.076 / 2 - 0.015],
            "orientation": [-90, 0, 0],
        },
        "passive_wheels": [
            {
                "name": "ZeroFrictionSphere",
                "radius": 0.076 / 2,
                "offset": [-0.24 / 2, 0.0, -0.420 / 2 + 0.076 / 2 - 0.015],
            },
            {
                "name": "ZeroFrictionSphere",
                "radius": 0.076 / 2,
                "offset": [0.24 / 2, 0.0, -0.420 / 2 + 0.076 / 2 - 0.015],
            },
        ],
    }

    AMR_2W_SS("/Turtlebot2", cfg={"system": Turtlebot2}, translation=[0, 0, 0.3])

    # Kobuki's Turtlebot 2
    Turtlebot2_caster = {
        "shape": {
            "name": "Cylinder",
            "radius": 0.354 / 2,
            "height": 0.420,
            "has_collider": True,
            "is_rigid": True,
            "refinement": 2,
        },
        "mass": 6.5,
        "left_wheel": {
            "wheel": {
                "visual_shape": {
                    "name": "Cylinder",
                    "radius": 0.076 / 2,
                    "height": 0.04,
                    "has_collider": False,
                    "is_rigid": False,
                    "refinement": 2,
                },
                "collider_shape": {
                    "name": "Capsule",
                    "radius": 0.076 / 2,
                    "height": 0.04,
                    "has_collider": True,
                    "is_rigid": True,
                    "refinement": 2,
                },
                "mass": 0.05,
            },
            "actuator": {
                "name": "RevoluteJoint",
                "axis": "Z",
                "enable_drive": True,
                "damping": 1e10,
                "stiffness": 0.0,
            },
            "offset": [0.0, -0.24 / 2, -0.420 / 2 + 0.076 / 2 - 0.015],
            "orientation": [-90, 0, 0],
        },
        "right_wheel": {
            "wheel": {
                "visual_shape": {
                    "name": "Cylinder",
                    "radius": 0.076 / 2,
                    "height": 0.04,
                    "has_collider": False,
                    "is_rigid": False,
                    "refinement": 2,
                },
                "collider_shape": {
                    "name": "Capsule",
                    "radius": 0.076 / 2,
                    "height": 0.04,
                    "has_collider": True,
                    "is_rigid": True,
                    "refinement": 2,
                },
                "mass": 0.05,
            },
            "actuator": {
                "name": "RevoluteJoint",
                "axis": "Z",
                "enable_drive": True,
                "damping": 1e10,
                "stiffness": 0.0,
            },
            "offset": [0.0, 0.24 / 2, -0.420 / 2 + 0.076 / 2 - 0.015],
            "orientation": [-90, 0, 0],
        },
        "passive_wheels": [
            {
                "name": "CasterWheel",
                "wheel": {
                    "visual_shape": {
                        "name": "Cylinder",
                        "radius": 0.076 / 2,
                        "height": 0.04,
                        "has_collider": False,
                        "is_rigid": False,
                        "refinement": 2,
                    },
                    "collider_shape": {
                        "name": "Capsule",
                        "radius": 0.076 / 2,
                        "height": 0.04,
                        "has_collider": True,
                        "is_rigid": True,
                        "refinement": 2,
                    },
                    "mass": 0.05,
                },
                "wheel_joint": {
                    "name": "RevoluteJoint",
                    "axis": "Z",
                    "enable_drive": False,
                },
                "caster_joint": {
                    "name": "RevoluteJoint",
                    "axis": "Z",
                    "enable_drive": False,
                },
                "caster_offset": [-0.24 / 2, 0.0, -0.420 / 2 + 0.076 - 0.015],
                "wheel_offset": [-0.24 / 2, 0.0, -0.420 / 2 + 0.076 / 2 - 0.015],
                "wheel_orientation": [-90, 0, 0],
            },
            {
                "name": "CasterWheel",
                "wheel": {
                    "visual_shape": {
                        "name": "Cylinder",
                        "radius": 0.076 / 2,
                        "height": 0.04,
                        "has_collider": False,
                        "is_rigid": False,
                        "refinement": 2,
                    },
                    "collider_shape": {
                        "name": "Capsule",
                        "radius": 0.076 / 2,
                        "height": 0.04,
                        "has_collider": True,
                        "is_rigid": True,
                        "refinement": 2,
                    },
                    "mass": 0.05,
                },
                "wheel_joint": {
                    "name": "RevoluteJoint",
                    "axis": "Z",
                    "enable_drive": False,
                },
                "caster_joint": {
                    "name": "RevoluteJoint",
                    "axis": "Z",
                    "enable_drive": False,
                },
                "caster_offset": [0.24 / 2, 0.0, -0.420 / 2 + 0.076 - 0.015],
                "wheel_offset": [0.24 / 2, 0.0, -0.420 / 2 + 0.076 / 2 - 0.015],
                "wheel_orientation": [-90, 0, 0],
            },
        ],
    }

    AMR_2W_SS(
        "/Turtlebot2_Caster",
        cfg={"system": Turtlebot2_caster},
        translation=[1.0, 0, 0.3],
    )

    world.reset()
    for i in range(100):
        world.step(render=True)

    timeline.play()
    while simulation_app.is_running():
        world.step(render=True)
    timeline.stop()
    simulation_app.close()