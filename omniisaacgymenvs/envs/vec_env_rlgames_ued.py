# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import numpy as np
from .vec_env_rlgames import VecEnvRLGames


# VecEnv Wrapper for UED training
class VecEnvRLGamesUED(VecEnvRLGames):
    def reset_random(self, rng):
		"""
		Reset the environment to a random level, i.e. using domain randomization, 
		and return the observation for the first time step in this level.
		"""
		raise NotImplementedError
	
	def reset_to_level(self, encoding):
		"""
		Reset the environment to a specific configuration defined by encoding.
		Encoding can be an array with last index being the rng seed.
		"""
		raise NotImplementedError
	
	def reset_student(self, rng, encoding):
		"""
		Reset the current environment level, i.e. only the student agent's state. 
		Do not change the actual environment configuration. 
		Returns the first observation in a new episode starting in that level.
		"""
		raise NotImplementedError

	@property
    def encoding(self):
		"""
		Returns the encoding of the current environment configuration.
		"""
		raise NotImplementedError
	
	@property
	def use_byte_encoding(self):
		"""
		Returns whether the encoding of the environment configuration is in bytes.
		"""
		raise NotImplementedError
	
	@property
	def adversary_action_space(self):
		"""
		Returns the action space of the adversary agent. E.g. gym.spaces.Box(...)
		"""
		raise NotImplementedError
	
	@property
	def adversary_observation_space(self):
		"""
		Returns the observation space of the adversary agent. E.g. gym.spaces.Dict(...)
		"""
		raise NotImplementedError
	

	def step_adversary(self, action):
		"""
		Step the adversary agent in the environment.
		"""
		raise NotImplementedError