import numpy as np
import unittest
import torch
import math

import omniisaacgymenvs.tasks.MFP.curriculum_helpers as ch

sigmoid_dict_0 = {"function": "sigmoid", "start": 0, "end": 1000, "extent": 1.5}
sigmoid_dict_1 = {"function": "sigmoid", "start": 100, "end": 1200, "extent": 3.0}
sigmoid_dict_2 = {"function": "sigmoid", "start": 200, "end": 1400, "extent": 4.5}
sigmoid_dict_3 = {"function": "sigmoid", "start": 400, "end": 1800, "extent": 6.0}
none_dict_0 = {"function": "none", "start": 0, "end": 1000}
none_dict_1 = {"function": "none", "start": 100, "end": 1200}
none_dict_2 = {"function": "none", "start": 200, "end": 1400}
none_dict_3 = {"function": "none", "start": 400, "end": 1800}
lin_dict_0 = {"function": "linear", "start": 0, "end": 1000}
lin_dict_1 = {"function": "linear", "start": 100, "end": 1200}
lin_dict_2 = {"function": "linear", "start": 200, "end": 1400}
lin_dict_3 = {"function": "linear", "start": 400, "end": 1800}
pow_dict_0 = {"function": "pow", "start": 0, "end": 1000, "alpha": 0.5}
pow_dict_1 = {"function": "pow", "start": 100, "end": 1200, "alpha": 0.75}
pow_dict_2 = {"function": "pow", "start": 200, "end": 1400, "alpha": 1.5}
pow_dict_3 = {"function": "pow", "start": 400, "end": 1800, "alpha": 3.0}

rate_list = [
    sigmoid_dict_0,
    sigmoid_dict_1,
    sigmoid_dict_2,
    sigmoid_dict_3,
    none_dict_0,
    none_dict_1,
    none_dict_2,
    none_dict_3,
    lin_dict_0,
    lin_dict_1,
    lin_dict_2,
    lin_dict_3,
    pow_dict_0,
    pow_dict_1,
    pow_dict_2,
    pow_dict_3,
]


trunc_norm_dict_0 = {
    "distribution": "truncated_normal",
    "start_mean": -0.5,
    "start_std": 0.5,
    "end_mean": 5.0,
    "end_std": 0.5,
    "min_value": -0.5,
    "max_value": 0.5,
}
trunc_norm_dict_1 = {
    "distribution": "truncated_normal",
    "start_mean": 0.0,
    "start_std": 0.01,
    "end_mean": 4.0,
    "end_std": 0.01,
    "min_value": 0.25,
    "max_value": 6.0,
}
trunc_norm_dict_2 = {
    "distribution": "truncated_normal",
    "start_mean": 0.25,
    "start_std": 0.5,
    "end_mean": 3.0,
    "end_std": 2.0,
    "min_value": 0.25,
    "max_value": 3.0,
}
trunc_norm_dict_3 = {
    "distribution": "truncated_normal",
    "start_mean": 0.5,
    "start_std": 0.5,
    "end_mean": 2.0,
    "end_std": 1.0,
    "min_value": 0.25,
    "max_value": 4.0,
}
norm_dict_0 = {
    "distribution": "normal",
    "start_mean": -0.5,
    "start_std": 0.5,
    "end_mean": 5.0,
    "end_std": 0.5,
}
norm_dict_1 = {
    "distribution": "normal",
    "start_mean": 0.0,
    "start_std": 0.01,
    "end_mean": 4.0,
    "end_std": 0.01,
}
norm_dict_2 = {
    "distribution": "normal",
    "start_mean": 0.25,
    "start_std": 0.5,
    "end_mean": 3.0,
    "end_std": 2.0,
}
norm_dict_3 = {
    "distribution": "normal",
    "start_mean": 0.5,
    "start_std": 0.5,
    "end_mean": 2.0,
    "end_std": 1.0,
}
uniform_dict_0 = {
    "distribution": "uniform",
    "start_min_value": -0.5,
    "start_max_value": 0.5,
    "end_min_value": 5.0,
    "end_max_value": 5.0,
}
uniform_dict_1 = {
    "distribution": "uniform",
    "start_min_value": 0.0,
    "start_max_value": 0.0,
    "end_min_value": 1.0,
    "end_max_value": 4.0,
}
uniform_dict_2 = {
    "distribution": "uniform",
    "start_min_value": 0.2,
    "start_max_value": 0.3,
    "end_min_value": 2.0,
    "end_max_value": 3.0,
}
uniform_dict_3 = {
    "distribution": "uniform",
    "start_min_value": 0.5,
    "start_max_value": 0.5,
    "end_min_value": -2.0,
    "end_max_value": 2.0,
}

dist_list = [
    trunc_norm_dict_0,
    trunc_norm_dict_1,
    trunc_norm_dict_2,
    trunc_norm_dict_3,
    norm_dict_0,
    norm_dict_1,
    norm_dict_2,
    norm_dict_3,
    uniform_dict_0,
    uniform_dict_1,
    uniform_dict_2,
    uniform_dict_3,
]


class TestCurriculumLoaders(unittest.TestCase):
    def test_loading_all_rate_loaders(self):
        success = False
        try:
            for rate in rate_list:
                ch.CurriculumRateParameters(**rate)
            success = True
        except:
            pass

        self.assertTrue(success)

    def test_all_sampler_loaders(self):
        success = False
        try:
            for dist in dist_list:
                ch.CurriculumSamplingParameters(**dist)
            success = True
        except:
            pass

        self.assertTrue(success)

    def test_sigmoid_rate_loader(self):
        rate = ch.CurriculumRateParameters(**sigmoid_dict_0)
        self.assertEqual(rate.function, ch.RateFunctionDict["sigmoid"])
        self.assertEqual(rate.start, 0)
        self.assertEqual(rate.end, 1000)
        self.assertEqual(rate.extent, 1.5)

    def test_none_rate_loader(self):
        rate = ch.CurriculumRateParameters(**none_dict_0)
        self.assertEqual(rate.function, ch.RateFunctionDict["none"])

    def test_linear_rate_loader(self):
        rate = ch.CurriculumRateParameters(**lin_dict_0)
        self.assertEqual(rate.function, ch.RateFunctionDict["linear"])
        self.assertEqual(rate.start, 0)
        self.assertEqual(rate.end, 1000)

    def test_pow_rate_loader(self):
        rate = ch.CurriculumRateParameters(**pow_dict_0)
        self.assertEqual(rate.function, ch.RateFunctionDict["pow"])
        self.assertEqual(rate.start, 0)
        self.assertEqual(rate.end, 1000)
        self.assertEqual(rate.alpha, 0.5)

    def test_error_handling_rate_loader(self):
        success = 1
        try:
            rate = ch.CurriculumRateParameters(
                **{"function": "none", "start": 0, "end": -1000}
            )
            success *= 0
        except:
            pass
        try:
            rate = ch.CurriculumRateParameters(
                **{"function": "none", "start": -100, "end": 1000}
            )
            success *= 0
        except:
            pass
        try:
            rate = ch.CurriculumRateParameters(
                **{"function": "sigmoid", "start": 100, "end": 1000, "extent": -1}
            )
            success *= 0
        except:
            pass
        try:
            rate = ch.CurriculumRateParameters(
                **{"function": "sigmoid", "start": 100, "end": 1000, "extent": 0}
            )
            success *= 0
        except:
            pass
        try:
            rate = ch.CurriculumRateParameters(
                **{"function": "linear", "start": 100, "end": -1000}
            )
            success *= 0
        except:
            pass
        try:
            rate = ch.CurriculumRateParameters(
                **{"function": "linear", "start": -1000, "end": -100}
            )
            success *= 0
        except:
            pass
        try:
            rate = ch.CurriculumRateParameters(
                **{"function": "pow", "start": 100, "end": 1000, "alpha": -1}
            )
            success *= 0
        except:
            pass

        self.assertTrue(success == 1)

    def test_load_empty_rate_loader(self):
        success = False
        try:
            rate = ch.CurriculumRateParameters(**{})
            success = True
        except:
            pass

        self.assertTrue(success)

    def test_load_empty_sampler_loader(self):
        success = False
        try:
            dist = ch.CurriculumSamplingParameters(**{})
            success = True
        except:
            pass

        self.assertTrue(success)

    def test_load_trunc_norm_sampler_loader(self):
        dist = ch.CurriculumSamplingParameters(**trunc_norm_dict_0)
        self.assertEqual(dist.function, ch.SampleFunctionDict["truncated_normal"])
        self.assertEqual(dist.start_mean, -0.5)
        self.assertEqual(dist.start_std, 0.5)
        self.assertEqual(dist.end_mean, 5.0)
        self.assertEqual(dist.end_std, 0.5)
        self.assertEqual(dist.min_value, -0.5)
        self.assertEqual(dist.max_value, 0.5)

    def test_load_norm_sampler_loader(self):
        dist = ch.CurriculumSamplingParameters(**norm_dict_0)
        self.assertEqual(dist.function, ch.SampleFunctionDict["normal"])
        self.assertEqual(dist.start_mean, -0.5)
        self.assertEqual(dist.start_std, 0.5)
        self.assertEqual(dist.end_mean, 5.0)
        self.assertEqual(dist.end_std, 0.5)

    def test_load_uniform_sampler_loader(self):
        dist = ch.CurriculumSamplingParameters(**uniform_dict_0)
        self.assertEqual(dist.function, ch.SampleFunctionDict["uniform"])
        self.assertEqual(dist.start_min_value, -0.5)
        self.assertEqual(dist.start_max_value, 0.5)
        self.assertEqual(dist.end_min_value, 5.0)
        self.assertEqual(dist.end_max_value, 5.0)


if __name__ == "__main__":
    unittest.main()
