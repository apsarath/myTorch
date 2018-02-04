import gym
from myTorch.environment import EnvironmentBase
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

class CartPoleImage(EnvironmentBase):

	def __init__(self, env_name):
		assert( env_name == "CartPole-v0" or env_name == "CartPole-v1")
		self._env_name = env_name
		self._env = gym.make(env_name).unwrapped
		self._screen_width = 600
		self._action_dim = self._env.action_space.n
		self._legal_moves = np.arange(self._action_dim)
		self._obs_dim = self._env.observation_space.shape
		self._resize = T.Compose([T.ToPILImage(), T.Scale(40, interpolation=Image.CUBIC), T.ToTensor()])

	def _get_cart_location(self):
		world_width = self._env.x_threshold * 2
		scale = self._screen_width / world_width
		return int(self._env.state[0] * scale + self._screen_width / 2.0)  # MIDDLE OF CART

	def _get_screen(self):
		screen = self._env.render(mode='rgb_array').transpose(
			(2, 0, 1))  # transpose into torch order (CHW)
		# Strip off the top and bottom of the screen
		screen = screen[:, 160:320]
		view_width = 320
		cart_location = self._get_cart_location()
		if cart_location < view_width // 2:
			slice_range = slice(view_width)
		elif cart_location > (self._screen_width - view_width // 2):
			slice_range = slice(-view_width, None)
		else:
			slice_range = slice(cart_location - view_width // 2,
								cart_location + view_width // 2)
		# Strip off the edges, so that we have a square image centered on a cart
		screen = screen[:, :, slice_range]
		# Convert to float, rescare, convert to torch tensor
		# (this doesn't require a copy)
		screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
		screen = torch.from_numpy(screen)
		# Resize, and add a batch dimension (BCHW)
		return self._resize(screen).numpy()


	@property
	def action_dim(self):
		return self._action_dim

	@property
	def obs_dim(self):
		return self._obs_dim

	def reset(self):
		obs = self._env.reset()
		last_screen = self._get_screen()
		current_screen = self._get_screen()
		obs = current_screen - last_screen
		self._last_screen = current_screen

		legal_moves = np.arange(self._action_dim)
		return obs, self._legal_moves

	def step(self, action):
		obs, reward, done, _ = self._env.step(action)
		
		current_screen = self._get_screen()
		obs = current_screen - self._last_screen
		self._last_screen = current_screen
		return obs, self._legal_moves, reward, done

	def render(self, mode='rgb_array'):
		pass

	def seed(self, seed):
		self._env._seed(seed=seed)

	def get_random_state(self):
		pass

	def set_random_state(self, state):
		pass

	def save(self, save_dir):
		return
		
	def load(self, save_dir):
		return  


if __name__=="__main__":
	env = CartPoleImage("CartPole-v0")
	env.reset()
	import pdb; pdb.set_trace()
