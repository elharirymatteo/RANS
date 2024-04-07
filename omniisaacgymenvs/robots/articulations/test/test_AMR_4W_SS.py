if __name__ == "__main__":
    from omni.isaac.kit import SimulationApp

    cfg = {
        "headless": False,
    }

    simulation_app = SimulationApp(cfg)
    from omni.isaac.core import World
    import omni

    from omniisaacgymenvs.robots.articulations.AMR_4WheelsSkidSteer import (
        AMR_4W_SS,
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

    # Clearpath Robotics' Husky
    Husky = {
        "shape": {
            "name": "Cube",
            "width": 0.670,
            "depth": 0.990,
            "height": 0.260,
            "has_collider": True,
            "is_rigid": True,
            "refinement": 2,
        },
        "mass": 50.0,
        "front_left_wheel": {
            "wheel": {
                "visual_shape": {
                    "name": "Cylinder",
                    "radius": 0.330 / 2,
                    "height": 0.125,
                    "has_collider": False,
                    "is_rigid": False,
                    "refinement": 2,
                },
                "collider_shape": {
                    "name": "Capsule",
                    "radius": 0.330 / 2,
                    "height": 0.125,
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
            "offset": [
                0.544 / 2,
                -0.670 / 2 - 0.125 / 2,
                -0.260 / 2 + 0.330 / 2 - 0.130,
            ],
            "orientation": [-90, 0, 0],
        },
        "front_right_wheel": {
            "wheel": {
                "visual_shape": {
                    "name": "Cylinder",
                    "radius": 0.330 / 2,
                    "height": 0.125,
                    "has_collider": False,
                    "is_rigid": False,
                    "refinement": 2,
                },
                "collider_shape": {
                    "name": "Capsule",
                    "radius": 0.330 / 2,
                    "height": 0.125,
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
            "offset": [
                0.544 / 2,
                0.670 / 2 + 0.125 / 2,
                -0.260 / 2 + 0.330 / 2 - 0.130,
            ],
            "orientation": [-90, 0, 0],
        },
        "rear_left_wheel": {
            "wheel": {
                "visual_shape": {
                    "name": "Cylinder",
                    "radius": 0.330 / 2,
                    "height": 0.125,
                    "has_collider": False,
                    "is_rigid": False,
                    "refinement": 2,
                },
                "collider_shape": {
                    "name": "Capsule",
                    "radius": 0.330 / 2,
                    "height": 0.125,
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
            "offset": [
                -0.544 / 2,
                -0.670 / 2 - 0.125 / 2,
                -0.260 / 2 + 0.330 / 2 - 0.130,
            ],
            "orientation": [-90, 0, 0],
        },
        "rear_right_wheel": {
            "wheel": {
                "visual_shape": {
                    "name": "Cylinder",
                    "radius": 0.330 / 2,
                    "height": 0.125,
                    "has_collider": False,
                    "is_rigid": False,
                    "refinement": 2,
                },
                "collider_shape": {
                    "name": "Capsule",
                    "radius": 0.330 / 2,
                    "height": 0.125,
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
            "offset": [
                -0.544 / 2,
                0.670 / 2 + 0.125 / 2,
                -0.260 / 2 + 0.330 / 2 - 0.130,
            ],
            "orientation": [-90, 0, 0],
        },
    }

    AMR_4W_SS("/Husky", cfg={"system": Husky}, translation=[0, 0, 0.3])

    world.reset()
    for i in range(100):
        world.step(render=True)

    timeline.play()
    while simulation_app.is_running():
        world.step(render=True)
    timeline.stop()
    simulation_app.close()
