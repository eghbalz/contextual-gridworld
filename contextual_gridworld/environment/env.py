
import gym
import os
import yaml

from contextual_gridworld.environment.colors import *
from enum import IntEnum
from gym import spaces
from gym.utils import seeding
from gym_minigrid.minigrid import DIR_TO_VEC
from gym_minigrid.register import register
from gym_minigrid.rendering import *


# Size in pixels of a tile in the full-scale human view
TILE_PIXELS = 32

# Map of object type to integers
OBJECT_TO_IDX = {
    'empty': 0,
    'wall': 1,
    'obstacle': 2,
    'goodie': 3,
    'goal': 4,
    'agent': 5
}

IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))


def load_reward_config(config_file: str) -> dict:
    """Load game config from YAML file."""
    with open(os.path.join(os.path.dirname(__file__), "reward_configurations", config_file), 'rb') as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)
    return config


def load_context_config(config_file: str) -> (dict, int):
    """Load game config from YAML file."""
    with open(os.path.join(os.path.dirname(__file__), "context_configurations", config_file), 'rb') as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)

    return config['contexts'], config['subdivs']


class WorldObj:
    """
    Base class for grid world objects
    """

    def __init__(self, type, color):
        assert type in OBJECT_TO_IDX, type
        assert color in COLOR_TO_IDX, color

        self.type = type
        self.color = color

        # Initial position of the object
        self.init_pos = None

        # Current position of the object
        self.cur_pos = None

    def encode(self):
        """Encode the a description of this object as a 3-tuple of integers"""
        return OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], 0

    def render(self, img):
        """Draw this object with the given renderer"""

        if self.type in ['goal', 'wall']:
            fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])
        elif self.type in ['obstacle', 'goodie']:
            fill_coords(img, point_in_circle(0.5, 0.5, 0.35), COLORS[self.color])
        else:
            raise NotImplementedError


class Grid:
    """
    Represent a grid and operations on it
    """

    # Static cache of pre-renderer tiles
    tile_cache = {}

    def __init__(self, width, height):
        assert width >= 3
        assert height >= 3

        self.width = width
        self.height = height

        self.grid = [None] * width * height

    def set(self, i, j, v):
        assert 0 <= i < self.width
        assert 0 <= j < self.height
        self.grid[j * self.width + i] = v

    def get(self, i, j):
        assert 0 <= i < self.width
        assert 0 <= j < self.height
        return self.grid[j * self.width + i]

    def horz_wall(self, x, y, length=None):
        if length is None:
            length = self.width - x
        for i in range(0, length):
            self.set(x + i, y, WorldObj('wall', 'grey'))

    def vert_wall(self, x, y, length=None):
        if length is None:
            length = self.height - y
        for j in range(0, length):
            self.set(x, y + j, WorldObj('wall', 'grey'))

    def wall_rect(self, x, y, w, h):
        self.horz_wall(x, y, w)
        self.horz_wall(x, y + h - 1, w)
        self.vert_wall(x, y, h)
        self.vert_wall(x + w - 1, y, h)

    def get_empty_positions(self):
        empty_positions = []
        for j in range(0, self.height):
            for i in range(0, self.width):
                cell = self.get(i, j)

                if cell is None:
                    empty_positions.append((i, j))
        return empty_positions

    @classmethod
    def render_tile(cls, obj, agent_dir=None, tile_size=TILE_PIXELS, subdivs=3, agent_color=None):
        """
        Render a tile and cache the result
        """

        # Hash map lookup key for the cache
        key = (agent_dir, tile_size)
        key = obj.encode() + key if obj else key

        key = (COLOR_TO_IDX[agent_color],) + key if agent_color is not None else key

        if key in cls.tile_cache:  # and agent_dir is None:
            return cls.tile_cache[key]

        img = np.zeros(shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8)

        if obj is not None:
            obj.render(img)

        if agent_dir is not None:
            tri_fn = point_in_triangle(
                (0.10, 0.19),
                (0.87, 0.50),
                (0.10, 0.81),
            )

            # Rotate the agent based on its direction
            tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5*math.pi*agent_dir)
            fill_coords(img, tri_fn, COLORS[agent_color])

        # Draw the grid lines (top and left edges)
        fill_coords(img, point_in_rect(0, 0.051, 0, 1), (100, 100, 100))
        fill_coords(img, point_in_rect(0, 1, 0, 0.051), (100, 100, 100))

        # Downsample the image to perform supersampling/anti-aliasing
        img = downsample(img, subdivs)

        # Cache the rendered tile
        cls.tile_cache[key] = img

        return img

    def render(self, tile_size, agent_pos=None, agent_dir=None, agent_id=None, subdivs=3):
        """
        Render this grid at a given scale
        """

        # Compute the total grid size
        width_px = self.width * tile_size
        height_px = self.height * tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        # Render the grid
        for j in range(0, self.height):
            for i in range(0, self.width):
                cell = self.get(i, j)

                # agent_here = np.array_equal(agent_pos, (i, j))
                agent_here = agent_pos[0] == i and agent_pos[1] == j
                tile_img = Grid.render_tile(
                    cell,
                    agent_dir=agent_dir if agent_here else None,
                    tile_size=tile_size,
                    agent_color=agent_id,
                    subdivs=subdivs
                )

                ymin = j * tile_size
                ymax = (j + 1) * tile_size
                xmin = i * tile_size
                xmax = (i + 1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img

        return img


class ContextualGridworldEnv(gym.Env):
    """
    2D contextual grid world game environment
    """

    metadata = {
        'render.modes': ['rgb_array'],
        'video.frames_per_second': 10
    }

    # Enumeration of possible actions
    class Actions(IntEnum):
        # Turn left, turn right, move forward
        left = 0
        right = 1
        forward = 2

    def __init__(
            self,
            grid_size=6,
            n_objects=4,
            max_steps=100,
            seed=1337,
            reward_config="default.yaml",
            context_config="reasoning_contexts_train.yaml",
            tile_size=8
    ):

        # Environment configuration
        self.tile_size = tile_size
        self.max_steps = max_steps

        self.width = grid_size
        self.height = grid_size

        # Action enumeration for this environment
        self.actions = ContextualGridworldEnv.Actions

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(self.actions))

        self.reward = load_reward_config(reward_config)

        self.contexts, self.subdivs = load_context_config(context_config)
        self.n_contexts = len(self.contexts)

        self.context_id = 0
        self.random_context = True
        self.context = self.contexts[self.context_id]

        self.context_encoding = np.zeros(len(self.contexts), dtype=np.float32)
        context_space = spaces.Box(low=0, high=1, shape=self.context_encoding.shape, dtype=np.float32)

        # Reduce obstacles if there are too many
        if n_objects <= grid_size / 2 + 1:
            self.n_obstacles = int(n_objects) // 2
            self.n_goodies = int(n_objects) // 2
        else:
            self.n_obstacles = int(grid_size / 2) // 2
            self.n_goodies = int(grid_size / 2) // 2

        self.obstacles = []
        self.goodies = []

        # Observations are dictionaries containing
        # a rgb representation of the grid and a one hot context representation

        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0, high=255, shape=(self.width*tile_size, self.height*tile_size, 3),
                                dtype=np.float32),
            'context': context_space,
        })

        # Range of possible rewards
        self.reward_range = (-1, 1)

        # Current position and direction of the agent
        self.agent_color = None
        self.agent_pos = None
        self.agent_dir = None

        self.goal_pos = None
        self.grid = None

        self.step_count = 0

        # random number generator
        self.np_random = None

        # Initialize the RNG
        self.seed(seed=seed)

        self.window = False
        # Initialize the state
        self.reset()

    def reset(self):
        # Current position and direction of the agent
        self.agent_pos = None
        self.agent_dir = None
        self.goal_pos = None

        # sample a random context
        if self.random_context:
            self.context_id = self.np_random.choice(len(self.contexts))

        self.context = self.contexts[self.context_id]

        self.agent_color = self.context['agent']

        # Create a new empty grid at the start of each episode
        self.grid = Grid(self.width, self.height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, self.width, self.height)

        # Place goal and agent in the grid
        self.place_goal_and_agent(self.width)

        # Place obstacles
        self.obstacles = []
        for i_obst in range(self.n_obstacles):
            self.obstacles.append(WorldObj('obstacle', self.context['obstacle']))
            self.place_obj(self.obstacles[i_obst])

        # place goodies
        self.goodies = []
        for i_obst in range(self.n_goodies):
            self.goodies.append(WorldObj('goodie', self.context['goodie']))
            self.place_obj(self.goodies[i_obst])

        assert self.agent_pos is not None
        assert self.agent_dir is not None
        assert self.goal_pos is not None

        # Check that the agent doesn't overlap with an object
        start_cell = self.grid.get(*self.agent_pos)
        assert start_cell is None

        # Step count since episode start
        self.step_count = 0

        # Return first observation
        obs = self.gen_obs()
        return obs

    def seed(self, seed=1337):
        # Seed the random number generator
        self.np_random, _ = seeding.np_random(seed)
        return [seed]

    def _rand_int(self, low, high):
        """
        Generate random integer in [low,high[
        """
        return self.np_random.randint(low, high)

    def place_obj(self, obj):
        """
        Place an object at an empty position in the grid
        """

        # get empty position
        empty_positions = self.grid.get_empty_positions()

        # filter agent position to not place the object where the agent is
        empty_positions = list(filter(lambda x: not (x[0] == self.agent_pos[0] and x[1] == self.agent_pos[1]),
                                      empty_positions))

        rand_idx = self._rand_int(0, len(empty_positions))
        pos = np.array(empty_positions[rand_idx])

        self.grid.set(*pos, obj)

        if obj is not None:
            obj.init_pos = pos
            obj.cur_pos = pos

        return pos

    def place_goal_and_agent(self, size=None):
        """
        Set the goal in one of the four cornes and place the agent accordingly
        """

        locs = [{'agent': (size - 2, size - 2), 'goal': (1, 1), 'agent_dir': 2},
                {'agent': (1, 1), 'goal': (size - 2, size - 2), 'agent_dir': 0},
                {'agent': (size - 2, 1), 'goal': (1, size - 2), 'agent_dir': 1},
                {'agent': (1, size - 2), 'goal': (size - 2, 1), 'agent_dir': 3}
                ]
        rand_ix = self._rand_int(0, 4)
        agent_goal_locs = locs[rand_ix]

        self.agent_pos = agent_goal_locs['agent']
        self.agent_dir = agent_goal_locs['agent_dir']
        self.goal_pos = agent_goal_locs['goal']

        self.grid.set(self.goal_pos[0], self.goal_pos[1], WorldObj('goal', self.context['goal']))

    @property
    def dir_vec(self):
        """
        Get the direction vector for the agent, pointing in the direction of forward movement.
        """
        assert 0 <= self.agent_dir < 4
        return DIR_TO_VEC[self.agent_dir]

    @property
    def front_pos(self):
        """
        Get the position of the cell that is right in front of the agent
        """
        return self.agent_pos + self.dir_vec

    def step(self, action):
        self.step_count += 1

        # default reward for each step
        reward = self.reward['stepwise']['reward']
        done = False
        goal_reached = 0

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell is None:
                self.agent_pos = fwd_pos

            elif fwd_cell.type == 'goal':
                self.agent_pos = fwd_pos
                done = True

                # reward for reaching the goal
                reward = self.reward['goal']['reward'] -\
                         self.reward['goal']['discounting'] * (self.step_count / self.max_steps)
                goal_reached = 1

        else:
            assert False, "unknown action"

        if self.step_count >= self.max_steps:
            done = True

        # goal not reached and episode is over
        if done and goal_reached == 0:
            # default set to 0 just in case its not in the config
            reward = self.reward['goal']['not_reached']

        # If the agent tried to walk into the wall or over an obstacle or goodie
        if not done and action == self.actions.forward and fwd_cell:
            # either wall, obstacle or goodie ahead
            reward = self.reward[fwd_cell.type]['reward']
            done = self.reward[fwd_cell.type]['reset']

            if fwd_cell.type == 'obstacle':
                # remove obstacle from grid
                self.obstacles.pop(self.obstacles.index(fwd_cell))
                self.grid.set(*fwd_cell.cur_pos, None)
            elif fwd_cell.type == 'goodie':
                # remove goodie from grid
                self.goodies.pop(self.goodies.index(fwd_cell))
                self.grid.set(*fwd_cell.cur_pos, None)

        obs = self.gen_obs()

        # useful logging information
        info = {'obstacles': len(self.obstacles),
                'goodies': len(self.goodies),
                'step_count': self.step_count,
                'context': self.context_id,
                'goal_reached': goal_reached}

        return obs, reward, done, info

    def gen_obs(self):
        """
        Generate the agent's view (fully observable rgb image)
        """

        # Encode the environment state into a numpy rgb array
        image = self.render('rgb_array', tile_size=self.tile_size)

        # rotate image based on agent direction
        image = np.rot90(image, k=self.agent_dir)

        context = self.context_encoding * 0
        context[self.context_id] = 1

        # Observations are dictionaries containing:
        # - an image (fully observable view of the environment encoded as an rgb image)
        # - one hot encoding of the current context
        obs = {'image': image, 'context': context}

        return obs

    def render(self, mode='rgb_array', close=False, tile_size=TILE_PIXELS):
        """
        Render the whole-grid human view
        """

        if close:
            if self.window:
                self.window.close()
            return

        if mode == 'human' and not self.window:
            import gym_minigrid.window
            self.window = gym_minigrid.window.Window('ContextualGridworld')
            self.window.show(block=False)

        # Render the whole grid
        img = self.grid.render(tile_size, self.agent_pos, self.agent_dir,
                               subdivs=self.subdivs, agent_id=self.agent_color)

        if mode == 'human':
            self.window.set_caption(f'Context: {self.context}')
            self.window.show_img(img)

        return img


register(
    id='MiniGrid-Contextual-v0',
    entry_point='contextual_gridworld.environment:ContextualGridworldEnv'
)
