if __name__ == "__main__":
    from omni.isaac.kit import SimulationApp

    cfg = {
        "headless": False,
    }

    simulation_app = SimulationApp(cfg)
    from omni.isaac.core import World
    from pxr import UsdLux
    import omni
    import yaml

    from robots.articulations.MFP2D_reaction_wheel import (
        ModularFloatingPlatform,
    )


    # Instantiate simulation
    timeline = omni.timeline.get_timeline_interface()
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    light = UsdLux.DistantLight.Define(world.stage, "/DistantLight")
    light.CreateIntensityAttr(3000.0)
    physics_ctx = world.get_physics_context()
    physics_ctx.set_solver_type("PGS")

    with open("cfg/task/robot/MFP2D_FP_reaction_wheel.yaml", "r") as file:
        reaction_wheel_cfg = yaml.safe_load(file)

    ModularFloatingPlatform("/MFP2D_FP_reaction_wheel", cfg=reaction_wheel_cfg, translation=[0, 0, 0.3])

    world.reset()
    for i in range(100):
        world.step(render=True)

    timeline.play()
    while simulation_app.is_running():
        world.step(render=True)
    timeline.stop()
    simulation_app.close()


    