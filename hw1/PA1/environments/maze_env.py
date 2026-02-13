"""
Custom Maze Environment for boundary-following and reward collection
"""

import numpy as np
from minigrid.core.grid import Grid
from minigrid.core.world_object import Wall, Ball, Goal, WorldObj
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.mission import MissionSpace
from minigrid.core.constants import COLOR_NAMES, OBJECT_TO_IDX, COLOR_TO_IDX


class Coin(Ball):
    """
    Collectible coin/reward object that can be walked over
    """
    def __init__(self, color='yellow'):
        super().__init__(color)
        
    def can_overlap(self):
        """Agent can walk over coins to collect them"""
        return True
    
    def can_pickup(self):
        """Coins are not picked up with pickup action"""
        return False


class MazeEnv(MiniGridEnv):
    """
    Custom maze environment with:
    - No narrow gaps (passages at least 2 cells wide)
    - Rewards placed along boundaries
    - 3x3 agent view window
    - Fixed step limit
    """
    
    def __init__(
        self,
        size=15,  # Grid size for square mazes (size x size)
        agent_view_size=3,
        max_steps=500,
        reward_type="outer",  # "outer", "inner", "both" - boundary reward placement
        walls_config=None,  # Custom walls configuration
        custom_cells=None,  # Custom cell modifications
        clear_areas=None,  # Rectangular areas to clear
        agent_start_pos=(2, 2),  # Agent starting position
        agent_start_dir=3,  # Agent starting direction (0=right, 1=down, 2=left, 3=up)
        **kwargs
    ):
        """
        Initialize the maze environment
        
        Args:
            size: Grid size for square mazes (size x size). Default is 15.
            agent_view_size: Size of agent's observation window
            max_steps: Maximum steps per episode
            reward_type: Where to place rewards ("inner", "outer", "both")
            walls_config: List of wall configurations. Each config is a dict with:
                         - 'type': 'horizontal' or 'vertical'
                         - 'x' or 'y': starting position
                         - 'start': start coordinate
                         - 'end': end coordinate
                         - 'gaps': list of gap ranges [(start, end), ...]
            custom_cells: List of custom cell modifications. Each is a dict with:
                         - 'x': x-coordinate
                         - 'y': y-coordinate
                         - 'type': 'wall', 'empty', or 'coin'
            clear_areas: List of rectangular areas to clear. Each is a dict with:
                        - 'x1': left x-coordinate
                        - 'y1': top y-coordinate
                        - 'x2': right x-coordinate
                        - 'y2': bottom y-coordinate
            agent_start_pos: Starting position (x, y) for the agent
            agent_start_dir: Starting direction for the agent (0=right, 1=down, 2=left, 3=up)
        """
        # Set maze dimensions (always square)
        self.maze_width = size
        self.maze_height = size
        
        self.reward_type = reward_type
        self.walls_config = walls_config or []
        self.custom_cells = custom_cells or []
        self.clear_areas = clear_areas or []
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.rewards_collected = 0
        self.total_rewards = 0
        
        # Define mission space
        mission_space = MissionSpace(
            mission_func=lambda: "Collect rewards along the boundaries!"
        )
        
        # Call parent constructor with width and height
        super().__init__(
            mission_space=mission_space,
            width=self.maze_width,
            height=self.maze_height,
            max_steps=max_steps,
            agent_view_size=agent_view_size,
            see_through_walls=False,  # Walls block vision
            **kwargs
        )
    
    def _gen_grid(self, width, height):
        """
        Generate the maze grid with no narrow gaps and rewards
        """
        # Create empty grid
        self.grid = Grid(width, height)
        
        # Generate walls (outer boundary)
        for i in range(width):
            self.grid.set(i, 0, Wall())
            self.grid.set(i, height - 1, Wall())
        for j in range(height):
            self.grid.set(0, j, Wall())
            self.grid.set(width - 1, j, Wall())
        
        # Add internal walls (ensuring no narrow gaps)
        # Simple maze pattern - you can make this more complex
        self._add_internal_walls(width, height)
        
        # Place agent at starting position
        self.agent_pos = self.agent_start_pos
        self.agent_dir = self.agent_start_dir
        
        # Place rewards
        self._place_rewards(width, height)
        
        # Apply custom cell modifications
        self._apply_custom_cells()
        
        # Apply clear areas
        self._apply_clear_areas()
        
        self.mission = f"Collect rewards! Progress: {self.rewards_collected}/{self.total_rewards}"
    
    def _add_internal_walls(self, width, height):
        """
        Add internal walls based on walls_config
        If no config provided, use default pattern
        """
        if not self.walls_config:
            # Default pattern (for backward compatibility)
            self._add_default_internal_walls(width, height)
        else:
            # Use custom walls configuration
            for wall_cfg in self.walls_config:
                self._add_wall_from_config(wall_cfg)
    
    def _add_default_internal_walls(self, width, height):
        """
        Add default internal walls pattern (original implementation)
        """
        # Vertical wall on the left side (with gaps)
        for j in range(3, height - 3):
            if j not in [6, 7, 8]:  # Leave a 3-cell gap
                self.grid.set(5, j, Wall())
        
        # Vertical wall on the right side (with gaps)
        for j in range(3, height - 3):
            if j not in [6, 7, 8]:
                self.grid.set(width - 6, j, Wall())
        
        # Horizontal wall (with gaps)
        for i in range(3, width - 3):
            if i not in [6, 7, 8, width - 7, width - 8, width - 9]:
                self.grid.set(i, height // 2, Wall())
        
        # Create a small room (ensuring 2-cell wide entrance)
        room_x, room_y = width // 2, height // 4
        room_size = 4
        for i in range(room_size):
            self.grid.set(room_x + i, room_y, Wall())
            self.grid.set(room_x + i, room_y + room_size - 1, Wall())
        for j in range(room_size):
            self.grid.set(room_x, room_y + j, Wall())
            self.grid.set(room_x + room_size - 1, room_y + j, Wall())
        # Create 2-cell wide entrance
        self.grid.set(room_x + 1, room_y, None)
        self.grid.set(room_x + 2, room_y, None)
    
    def _add_wall_from_config(self, config):
        """
        Add a wall segment based on configuration
        
        Args:
            config: dict with wall specification
                - 'type': 'horizontal' or 'vertical'
                - 'x' or 'y': fixed coordinate
                - 'start': start of the line
                - 'end': end of the line
                - 'gaps': optional list of gap ranges [(start, end), ...]
        """
        wall_type = config.get('type')
        gaps = config.get('gaps', [])
        
        if wall_type == 'horizontal':
            y = config['y']
            start_x = config['start']
            end_x = config['end']
            
            for x in range(start_x, end_x + 1):
                # Check if this position is in a gap
                in_gap = False
                for gap_start, gap_end in gaps:
                    if gap_start <= x <= gap_end:
                        in_gap = True
                        break
                
                if not in_gap and self.grid.get(x, y) is None:
                    self.grid.set(x, y, Wall())
        
        elif wall_type == 'vertical':
            x = config['x']
            start_y = config['start']
            end_y = config['end']
            
            for y in range(start_y, end_y + 1):
                # Check if this position is in a gap
                in_gap = False
                for gap_start, gap_end in gaps:
                    if gap_start <= y <= gap_end:
                        in_gap = True
                        break
                
                if not in_gap and self.grid.get(x, y) is None:
                    self.grid.set(x, y, Wall())
    
    def _place_rewards(self, width, height):
        """
        Place reward objects along wall boundaries
        """
        rewards_placed = 0
        
        if self.reward_type == "outer":
            # Place rewards along outer boundary only
            rewards_placed += self._place_outer_boundary_rewards(width, height)
        elif self.reward_type == "inner":
            # Place rewards along inner walls only
            rewards_placed += self._place_inner_boundary_rewards(width, height)
        elif self.reward_type == "both":
            # Place rewards along both outer and inner boundaries
            rewards_placed += self._place_outer_boundary_rewards(width, height)
            rewards_placed += self._place_inner_boundary_rewards(width, height)
        else:
            # Default: both boundaries
            rewards_placed += self._place_outer_boundary_rewards(width, height)
            rewards_placed += self._place_inner_boundary_rewards(width, height)
        
        self.total_rewards = rewards_placed
    
    def _place_outer_boundary_rewards(self, width, height):
        """Place rewards consecutively along outer boundaries"""
        rewards_placed = 0
        
        # Top boundary (consecutive, every cell)
        for i in range(1, width - 1):
            if self.grid.get(i, 1) is None:
                self.grid.set(i, 1, Coin(color='yellow'))
                rewards_placed += 1
        
        # Bottom boundary (consecutive, every cell)
        for i in range(1, width - 1):
            if self.grid.get(i, height - 2) is None:
                self.grid.set(i, height - 2, Coin(color='yellow'))
                rewards_placed += 1
        
        # Left boundary (consecutive, every cell, excluding corners)
        for j in range(2, height - 2):
            if self.grid.get(1, j) is None:
                self.grid.set(1, j, Coin(color='yellow'))
                rewards_placed += 1
        
        # Right boundary (consecutive, every cell, excluding corners)
        for j in range(2, height - 2):
            if self.grid.get(width - 2, j) is None:
                self.grid.set(width - 2, j, Coin(color='yellow'))
                rewards_placed += 1
        
        return rewards_placed
    
    def _place_inner_boundary_rewards(self, width, height):
        """Place rewards consecutively along inner wall boundaries"""
        rewards_placed = 0
        
        # Scan all cells and place rewards next to walls (not on outer boundary)
        for i in range(2, width - 2):
            for j in range(2, height - 2):
                cell = self.grid.get(i, j)
                # If cell is empty and next to a wall
                if cell is None and self._is_next_to_wall(i, j):
                    self.grid.set(i, j, Coin(color='yellow'))
                    rewards_placed += 1
        
        return rewards_placed
    
    def _is_next_to_wall(self, x, y):
        """Check if position (x, y) is adjacent to a wall"""
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if isinstance(self.grid.get(nx, ny), Wall):
                return True
        return False
    
    def _apply_custom_cells(self):
        """
        Apply custom cell modifications from custom_cells config
        """
        for cell_cfg in self.custom_cells:
            x = cell_cfg['x']
            y = cell_cfg['y']
            cell_type = cell_cfg['type']
            
            if cell_type == 'wall':
                self.grid.set(x, y, Wall())
            elif cell_type == 'empty':
                # If removing a coin, decrease total_rewards
                if isinstance(self.grid.get(x, y), Coin):
                    self.total_rewards -= 1
                self.grid.set(x, y, None)
            elif cell_type == 'coin':
                # If adding a coin, increase total_rewards
                if not isinstance(self.grid.get(x, y), Coin):
                    self.total_rewards += 1
                self.grid.set(x, y, Coin(color='yellow'))
    
    def clear_rectangle(self, x1, y1, x2, y2):
        """
        Clear a rectangular area to empty (removes walls/coins)
        
        Args:
            x1: Left x coordinate (top-left corner)
            y1: Top y coordinate (top-left corner)
            x2: Right x coordinate (bottom-right corner)
            y2: Bottom y coordinate (bottom-right corner)
        """
        for x in range(x1, x2 + 1):
            for y in range(y1, y2 + 1):
                cell = self.grid.get(x, y)
                if isinstance(cell, Coin):
                    self.total_rewards -= 1
                self.grid.set(x, y, None)
    
    def _apply_clear_areas(self):
        """
        Apply rectangular area clearing from clear_areas config
        """
        for area in self.clear_areas:
            x1 = area['x1']
            y1 = area['y1']
            x2 = area['x2']
            y2 = area['y2']
            self.clear_rectangle(x1, y1, x2, y2)
    
    def step(self, action):
        """
        Execute action and return observation
        
        Args:
            action: Action to take (0=left, 1=right, 2=forward)
        
        Returns:
            obs, reward, terminated, truncated, info
        """
        # Store position before action
        old_pos = self.agent_pos
        
        # Execute action
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Check if agent picked up a reward
        cell = self.grid.get(*self.agent_pos)
        if isinstance(cell, Coin):
            # Collect reward
            self.grid.set(*self.agent_pos, None)
            reward = 1.0
            self.rewards_collected += 1
            info['reward_collected'] = True
        else:
            info['reward_collected'] = False
        
        # Update mission to show progress
        self.mission = f"Collect rewards! Progress: {self.rewards_collected}/{self.total_rewards}"
        
        # Update info
        info['rewards_collected'] = self.rewards_collected
        info['total_rewards'] = self.total_rewards
        info['collection_rate'] = self.rewards_collected / self.total_rewards if self.total_rewards > 0 else 0
        
        # Check if all rewards collected
        if self.rewards_collected >= self.total_rewards:
            terminated = True
            info['success'] = True
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, seed=None, **kwargs):
        """Reset the environment"""
        # Set seed for reproducibility if provided
        if seed is not None:
            super().reset(seed=seed, **kwargs)
        else:
            super().reset(**kwargs)
        
        # Ensure agent starts at configured position and direction
        self.agent_pos = self.agent_start_pos
        self.agent_dir = self.agent_start_dir
        
        self.rewards_collected = 0
        
        # Return observation
        obs = self.gen_obs()
        info = {
            'rewards_collected': self.rewards_collected,
            'total_rewards': self.total_rewards,
        }
        return obs, info
    
    def gen_obs_grid(self, agent_view_size=None):
        """
        Generate the sub-grid observed by the agent with agent centered and rotated.
        Creates an agent-centric view where forward is always at the top (y=0).
        """
        if agent_view_size is None:
            agent_view_size = self.agent_view_size
        
        # Get agent position and direction
        ax, ay = self.agent_pos
        agent_dir = self.agent_dir
        
        # Create centered grid (agent at center)
        grid = Grid(agent_view_size, agent_view_size)
        
        # Create visibility mask (all cells visible in centered view)
        vis_mask = np.ones((agent_view_size, agent_view_size), dtype=bool)
        
        # Calculate offset to center the agent
        offset = agent_view_size // 2
        
        # Direction vectors for each orientation (dx, dy)
        # 0=right, 1=down, 2=left, 3=up
        dir_vec = [
            (1, 0),   # right: forward is +x
            (0, 1),   # down: forward is +y
            (-1, 0),  # left: forward is -x
            (0, -1)   # up: forward is -y
        ]
        
        # Right vector (perpendicular to forward, 90° clockwise)
        right_vec = [
            (0, 1),   # when facing right, right is down (+y)
            (-1, 0),  # when facing down, right is left (-x)
            (0, -1),  # when facing left, right is up (-y)
            (1, 0)    # when facing up, right is right (+x)
        ]
        
        # Get direction vectors for current orientation
        dx_forward, dy_forward = dir_vec[agent_dir]
        dx_right, dy_right = right_vec[agent_dir]
        
        # Fill the grid with agent-centric coordinates
        # In the view: x increases to the right, y increases forward (up in view)
        for view_x in range(agent_view_size):
            for view_y in range(agent_view_size):
                # Calculate relative position from agent's perspective
                # view_y: 0 = top (forward), 2 = bottom (behind)
                # view_x: 0 = left, 2 = right
                rel_right = view_x - offset  # negative = left, positive = right
                rel_forward = offset - view_y  # positive = forward, negative = behind
                
                # Transform to world coordinates based on agent orientation
                world_x = ax + (rel_forward * dx_forward) + (rel_right * dx_right)
                world_y = ay + (rel_forward * dy_forward) + (rel_right * dy_right)
                
                # Check if within bounds
                if 0 <= world_x < self.width and 0 <= world_y < self.height:
                    # Don't include the agent itself
                    if (world_x, world_y) != self.agent_pos:
                        cell = self.grid.get(world_x, world_y)
                        if cell is not None:
                            grid.set(view_x, view_y, cell)
                else:
                    # Out of bounds - treat as wall
                    grid.set(view_x, view_y, Wall())
        
        return grid, vis_mask
    
    def gen_obs(self):
        """
        Generate agent's observation with agent centered in view
        """
        grid, _ = self.gen_obs_grid()
        
        # Generate the observation image
        image = grid.encode()
        
        # Get the center position
        center = self.agent_view_size // 2
        
        # Add agent at center with its direction
        image[center, center, :] = np.array([
            OBJECT_TO_IDX['agent'],
            COLOR_TO_IDX['red'],
            self.agent_dir
        ])
        
        obs = {
            'image': image,
            'direction': self.agent_dir,
            'mission': self.mission
        }
        
        return obs
    
    def get_pov_render(self, tile_size):
        """
        Render the agent's point of view (centered observation)
        Agent always faces up (direction 3) in this view
        """
        obs = self.gen_obs()
        grid, _ = self.gen_obs_grid()
        
        # Render the centered grid with agent always facing up
        img = grid.render(
            tile_size,
            agent_pos=(self.agent_view_size // 2, self.agent_view_size // 2),
            agent_dir=3,  # Always show agent facing up in POV
            highlight_mask=None
        )
        
        return img
    
    def get_full_render(self, highlight, tile_size):
        """
        Render the full environment with centered observation window highlighted
        Override to show centered observation window instead of forward-facing
        """
        # If highlighting the observation, use centered window
        if highlight:
            # Get the centered observation grid and mask
            obs_grid, vis_mask = self.gen_obs_grid()
            
            # Calculate the top-left corner of the centered view in world coordinates
            offset = self.agent_view_size // 2
            top_left_x = self.agent_pos[0] - offset
            top_left_y = self.agent_pos[1] - offset
            
            # Create a mask for highlighting the centered observation area
            highlight_mask = np.zeros((self.width, self.height), dtype=bool)
            
            for view_x in range(self.agent_view_size):
                for view_y in range(self.agent_view_size):
                    world_x = top_left_x + view_x
                    world_y = top_left_y + view_y
                    
                    if 0 <= world_x < self.width and 0 <= world_y < self.height:
                        highlight_mask[world_x, world_y] = True
            
            # Render with our custom highlight mask
            img = self.grid.render(
                tile_size,
                self.agent_pos,
                self.agent_dir,
                highlight_mask=highlight_mask
            )
        else:
            # Render without highlighting
            img = self.grid.render(
                tile_size,
                self.agent_pos,
                self.agent_dir,
                highlight_mask=None
            )
        
        return img


class SimpleMazeEnv(MazeEnv):
    """Simple 15x15 maze with no inner walls, rewards only on outer boundary"""
    def __init__(self, **kwargs):
        walls = [
            {'type': 'vertical', 'x': 4, 'start': 3, 'end': 11, 'gaps': [(5, 9)]},
            {'type': 'vertical', 'x': 8, 'start': 3, 'end': 11},
            {'type': 'vertical', 'x': 12, 'start': 1, 'end': 3},
            {'type': 'vertical', 'x': 13, 'start': 1, 'end': 3},
            {'type': 'horizontal', 'y': 7, 'start': 3, 'end': 11, 'gaps': [(6, 8)]},
            {'type': 'horizontal', 'y': 11, 'start': 9, 'end': 11},
        ]

        custom_cells = [
            {'x': 11, 'y': 2, 'type': 'coin'},
            {'x': 11, 'y': 3, 'type': 'coin'},
            {'x': 11, 'y': 4, 'type': 'coin'},
            {'x': 12, 'y': 4, 'type': 'coin'},
            {'x': 13, 'y': 4, 'type': 'coin'},
        ]

        super().__init__(size=15, reward_type="outer", walls_config=walls, custom_cells=custom_cells, max_steps=60, **kwargs)


class MediumMazeEnv(MazeEnv):
    """Medium 27x27 maze with custom walls"""
    def __init__(self, **kwargs):
        walls = [
            {'type': 'horizontal', 'y': 1, 'start': 1, 'end': 25},
            {'type': 'horizontal', 'y': 2, 'start': 1, 'end': 25},
            {'type': 'horizontal', 'y': 3, 'start': 1, 'end': 25},

            {'type': 'horizontal', 'y': 23, 'start': 1, 'end': 25},
            {'type': 'horizontal', 'y': 24, 'start': 1, 'end': 25},
            {'type': 'horizontal', 'y': 25, 'start': 1, 'end': 25},

            {'type': 'vertical', 'x': 7, 'start': 6, 'end': 20},
            {'type': 'vertical', 'x': 10, 'start': 6, 'end': 20, 'gaps': [(7, 12)]},
            {'type': 'vertical', 'x': 14, 'start': 6, 'end': 20, 'gaps': [(14, 19)]},
            {'type': 'vertical', 'x': 18, 'start': 6, 'end': 20},
            {'type': 'vertical', 'x': 23, 'start': 6, 'end': 20},
            {'type': 'horizontal', 'y': 6, 'start': 3, 'end': 13, 'gaps': [(8, 10)]},
            {'type': 'horizontal', 'y': 13, 'start': 3, 'end': 13, 'gaps': [(8, 10)]},
            {'type': 'horizontal', 'y': 20, 'start': 3, 'end': 23},

        ]

        clear_areas = [
            {'x1': 1, 'y1': 4, 'x2': 24, 'y2': 4},
            {'x1': 1, 'y1': 22, 'x2': 24, 'y2': 22},
        ]

        super().__init__(size=27,
                         reward_type="inner",
                         walls_config=walls,
                         clear_areas=clear_areas,
                         max_steps=250,
                         **kwargs
        )


class HardMazeEnv(MazeEnv):
    """Hard 32x32 maze with custom walls"""
    def __init__(self, **kwargs):
        walls = [
            {'type': 'horizontal', 'y': 1, 'start': 1, 'end': 31},
            {'type': 'horizontal', 'y': 2, 'start': 1, 'end': 31},
            {'type': 'horizontal', 'y': 3, 'start': 1, 'end': 31},
            {'type': 'horizontal', 'y': 4, 'start': 1, 'end': 31},
            {'type': 'horizontal', 'y': 5, 'start': 1, 'end': 31},
            {'type': 'horizontal', 'y': 26, 'start': 1, 'end': 31},
            {'type': 'horizontal', 'y': 27, 'start': 1, 'end': 31},
            {'type': 'horizontal', 'y': 28, 'start': 1, 'end': 31},
            {'type': 'horizontal', 'y': 29, 'start': 1, 'end': 31},
            {'type': 'horizontal', 'y': 30, 'start': 1, 'end': 31},
            
            {'type': 'horizontal', 'y': 8, 'start': 8, 'end': 27, 'gaps': [(11, 20)]},
            {'type': 'horizontal', 'y': 9, 'start': 8, 'end': 25, 'gaps': [(11, 22)]},
            {'type': 'horizontal', 'y': 10, 'start': 7, 'end': 25, 'gaps': [(12, 22)]},
            {'type': 'horizontal', 'y': 11, 'start': 7, 'end': 25, 'gaps': [(8, 9), (13, 22)]},
            {'type': 'horizontal', 'y': 12, 'start': 7, 'end': 25, 'gaps': [(8, 9), (13, 22)]},
            {'type': 'horizontal', 'y': 13, 'start': 6, 'end': 25, 'gaps': [(8, 9), (13, 22)]},
            {'type': 'horizontal', 'y': 14, 'start': 6, 'end': 25, 'gaps': [(7, 10), (14, 22)]},
            {'type': 'horizontal', 'y': 15, 'start': 6, 'end': 25, 'gaps': [(7, 11), (14, 22)]},
            {'type': 'horizontal', 'y': 16, 'start': 6, 'end': 25, 'gaps': [(7, 11), (15, 22)]},
            {'type': 'horizontal', 'y': 17, 'start': 6, 'end': 25, 'gaps': [(15, 22)]},
            {'type': 'horizontal', 'y': 18, 'start': 5, 'end': 25, 'gaps': [(7, 11), (16, 22)]},
            {'type': 'horizontal', 'y': 19, 'start': 5, 'end': 25, 'gaps': [(7, 12), (16, 22)]},
            {'type': 'horizontal', 'y': 20, 'start': 5, 'end': 25, 'gaps': [(7, 12), (17, 22)]},
            {'type': 'horizontal', 'y': 21, 'start': 5, 'end': 25, 'gaps': [(7, 13), (17, 22)]},
            {'type': 'horizontal', 'y': 22, 'start': 4, 'end': 25, 'gaps': [(7, 14), (18, 22)]},
            {'type': 'horizontal', 'y': 23, 'start': 3, 'end': 27, 'gaps': [(8, 14)]},
        ]

        clear_areas = [
            {'x1': 1, 'y1': 6, 'x2': 30, 'y2': 6},
            {'x1': 1, 'y1': 25, 'x2': 30, 'y2': 25},
            {'x1': 8, 'y1': 11, 'x2': 9, 'y2': 13},
            {'x1': 7, 'y1': 14, 'x2': 10, 'y2': 14},
            {'x1': 7, 'y1': 15, 'x2': 11, 'y2': 16},
        ]

        super().__init__(size=32,
                         reward_type="inner",
                         walls_config=walls,
                         clear_areas=clear_areas,
                         max_steps=200,
                         **kwargs
        )