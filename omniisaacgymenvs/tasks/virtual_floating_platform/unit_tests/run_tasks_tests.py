from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp()

import unittest

testmodules = [
    "omniisaacgymenvs.tasks.virtual_floating_platform.unit_tests.test_MFP2D_disturbances",
    "omniisaacgymenvs.tasks.virtual_floating_platform.unit_tests.test_MFP2D_parameters",
    "omniisaacgymenvs.tasks.virtual_floating_platform.unit_tests.test_MFP2D_rewards",
    "omniisaacgymenvs.tasks.virtual_floating_platform.unit_tests.test_MFP2D_core",
    "omniisaacgymenvs.tasks.virtual_floating_platform.unit_tests.test_MFP2D_go_to_xy",
    "omniisaacgymenvs.tasks.virtual_floating_platform.unit_tests.test_MFP2D_go_to_pose",
    "omniisaacgymenvs.tasks.virtual_floating_platform.unit_tests.test_MFP2D_track_xy_vel",
    "omniisaacgymenvs.tasks.virtual_floating_platform.unit_tests.test_MFP2D_track_xyo_vel",
    "omniisaacgymenvs.tasks.virtual_floating_platform.unit_tests.test_MFP3D_disturbances",
    "omniisaacgymenvs.tasks.virtual_floating_platform.unit_tests.test_MFP3D_parameters",
    "omniisaacgymenvs.tasks.virtual_floating_platform.unit_tests.test_MFP3D_rewards",
    "omniisaacgymenvs.tasks.virtual_floating_platform.unit_tests.test_MFP3D_core",
    "omniisaacgymenvs.tasks.virtual_floating_platform.unit_tests.test_MFP3D_go_to_xyz",
    "omniisaacgymenvs.tasks.virtual_floating_platform.unit_tests.test_MFP3D_go_to_pose",
    "omniisaacgymenvs.tasks.virtual_floating_platform.unit_tests.test_MFP3D_track_xyz_vel",
    "omniisaacgymenvs.tasks.virtual_floating_platform.unit_tests.test_MFP3D_track_6d_vel",
]

suite = unittest.TestSuite()

for t in testmodules:
    try:
        # If the module defines a suite() function, call it to get the suite.
        mod = __import__(t, globals(), locals(), ["suite"])
        suitefn = getattr(mod, "suite")
        suite.addTest(suitefn())
    except (ImportError, AttributeError):
        # else, just load all the test cases from the module.
        suite.addTest(unittest.defaultTestLoader.loadTestsFromName(t))

unittest.TextTestRunner(verbosity=2).run(suite)

simulation_app.close()
