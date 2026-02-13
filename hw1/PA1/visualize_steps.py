"""
Step-by-Step Visualization Script for Agent Analysis

This script allows you to:
1. Run an agent and record all steps
2. Step through the episode using left/right arrow keys
3. View agent's observation, decision reasoning, and performance
4. Identify potential errors in agent behavior

Controls:
- Left Arrow: Previous step
- Right Arrow: Next step
- Space: Play/Pause auto-play
- R: Reset to beginning
- Q/ESC: Quit
- 1-9: Set playback speed
"""

import sys
import pygame
import numpy as np
from environments.maze_env import SimpleMazeEnv, MediumMazeEnv, HardMazeEnv
from agents import NaiveAgent, ProductionRulesAgent, StateMachineAgent


class StepVisualizer:
    """Visualize agent behavior step-by-step with playback controls"""
    
    def __init__(self, env, agent, max_steps=1000):
        """
        Initialize the visualizer
        
        Args:
            env: The maze environment
            agent: The agent to visualize
            max_steps: Maximum steps to record
        """
        self.env = env
        self.agent = agent
        self.max_steps = max_steps
        
        # Recorded episode data
        self.steps = []  # List of (obs, action, reward, done, info)
        self.current_step = 0
        self.total_steps = 0
        
        # Pygame setup
        pygame.init()
        self.tile_size = 32
        self.window_width = 1400
        self.window_height = 1000
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Agent Step-by-Step Visualizer")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 20)
        
        # Playback control
        self.playing = False
        self.playback_speed = 5  # Steps per second
        self.frame_counter = 0
        
    def record_episode(self):
        """Record a complete episode from the agent"""
        print("Recording episode...")
        obs, info = self.env.reset()
        self.agent.reset()
        
        self.steps = []
        done = False
        total_reward = 0
        step_count = 0
        collected_coins = set()  # Track which coin positions have been collected
        
        # Record initial state (before any action)
        self.steps.append({
            'step': -1,  # Initial state before first action
            'obs': obs.copy(),
            'action': None,  # No action taken yet
            'reward': 0,
            'done': False,
            'info': info.copy(),
            'agent_pos': self.env.agent_pos,
            'agent_dir': self.env.agent_dir,
            'total_reward': 0,
            'collected_coins': collected_coins.copy()
        })
        
        while not done and step_count < self.max_steps:
            # Get agent's perception and decision
            self.agent.perceive(obs)
            action = self.agent.decide()
            
            # Take action
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            # Track collected coins
            if info.get('reward_collected', False):
                collected_coins.add(self.env.agent_pos)
            
            # Record step
            self.steps.append({
                'step': step_count,
                'obs': obs.copy(),
                'action': action,
                'reward': reward,
                'done': done,
                'info': info.copy(),
                'agent_pos': self.env.agent_pos,
                'agent_dir': self.env.agent_dir,
                'total_reward': total_reward,
                'collected_coins': collected_coins.copy()  # Store which coins have been collected up to this point
            })
            
            obs = next_obs
            step_count += 1
            
            self.agent.update_stats(reward, done)
        
        self.total_steps = len(self.steps)
        print(f"Episode recorded: {self.total_steps - 1} steps, Total reward: {total_reward}")
        
        # Reset environment to initial state for visualization
        self.env.reset()
        if self.total_steps > 0:
            self._restore_state(0)
    
    def _restore_state(self, step_idx):
        """Restore environment to a specific step"""
        if 0 <= step_idx < self.total_steps:
            step_data = self.steps[step_idx]
            
            # First, reset environment to get fresh grid with all coins
            self.env.reset()
            
            # Set agent position and direction
            self.env.agent_pos = step_data['agent_pos']
            self.env.agent_dir = step_data['agent_dir']
            
            # Remove coins that have been collected up to this point
            from environments.maze_env import Coin
            collected_coins = step_data.get('collected_coins', set())
            for coin_pos in collected_coins:
                cell = self.env.grid.get(*coin_pos)
                if isinstance(cell, Coin):
                    self.env.grid.set(*coin_pos, None)
            
            # Update rewards collected count
            self.env.rewards_collected = len(collected_coins)
            
            # Update agent perception
            self.agent.perceive(step_data['obs'])
    
    def run(self):
        """Run the visualization loop"""
        if self.total_steps == 0:
            print("No episode recorded. Recording now...")
            self.record_episode()
        
        if self.total_steps == 0:
            print("Failed to record episode.")
            return
        
        running = True
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.KEYDOWN:
                    if event.key in [pygame.K_q, pygame.K_ESCAPE]:
                        running = False
                    
                    elif event.key == pygame.K_LEFT:
                        # Previous step
                        self.current_step = max(0, self.current_step - 1)
                        self._restore_state(self.current_step)
                        self.playing = False
                    
                    elif event.key == pygame.K_RIGHT:
                        # Next step
                        self.current_step = min(self.total_steps - 1, self.current_step + 1)
                        self._restore_state(self.current_step)
                        self.playing = False
                    
                    elif event.key == pygame.K_SPACE:
                        # Toggle play/pause
                        self.playing = not self.playing
                    
                    elif event.key == pygame.K_r:
                        # Reset to beginning
                        self.current_step = 0
                        self._restore_state(self.current_step)
                        self.playing = False
                    
                    elif event.key in [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, 
                                      pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8, pygame.K_9]:
                        # Set playback speed
                        speed_map = {
                            pygame.K_1: 1, pygame.K_2: 2, pygame.K_3: 3,
                            pygame.K_4: 5, pygame.K_5: 10, pygame.K_6: 15,
                            pygame.K_7: 20, pygame.K_8: 30, pygame.K_9: 60
                        }
                        self.playback_speed = speed_map[event.key]
            
            # Auto-play
            if self.playing:
                self.frame_counter += 1
                frames_per_step = max(1, 60 // self.playback_speed)
                
                if self.frame_counter >= frames_per_step:
                    self.frame_counter = 0
                    self.current_step += 1
                    
                    if self.current_step >= self.total_steps:
                        self.current_step = self.total_steps - 1
                        self.playing = False
                    else:
                        self._restore_state(self.current_step)
            
            # Render
            self.render()
            self.clock.tick(60)
        
        pygame.quit()
    
    def render(self):
        """Render the current state"""
        self.screen.fill((40, 40, 40))
        
        if self.total_steps == 0:
            # No data to display
            text = self.font.render("No episode data", True, (255, 255, 255))
            self.screen.blit(text, (50, 50))
            pygame.display.flip()
            return
        
        step_data = self.steps[self.current_step]
        
        # Render full maze view (left side)
        full_render = self.env.get_full_render(highlight=True, tile_size=self.tile_size)
        full_surface = pygame.surfarray.make_surface(np.transpose(full_render, (1, 0, 2)))
        self.screen.blit(full_surface, (20, 20))
        
        # Render agent's POV (right side, top)
        pov_render = self.env.get_pov_render(tile_size=64)
        pov_surface = pygame.surfarray.make_surface(np.transpose(pov_render, (1, 0, 2)))
        pov_x = full_render.shape[0] + 60
        self.screen.blit(pov_surface, (pov_x, 20))
        
        # Render observation grid visualization
        obs_y = 20 + pov_render.shape[1] + 40
        self._render_observation_grid(step_data['obs'], pov_x, obs_y)
        
        # Render step information (right side, bottom)
        info_x = pov_x
        info_y = obs_y + 250
        self._render_step_info(step_data, info_x, info_y)
        
        # Render controls (bottom)
        self._render_controls()
        
        pygame.display.flip()
    
    def _render_observation_grid(self, obs, x, y):
        """Render the 3x3 observation grid with labels"""
        # Get the observation grid directly (same as used for POV render)
        obs_grid, _ = self.env.gen_obs_grid()
        
        # Check if agent is StateMachineAgent (limited perception)
        is_limited_perception = isinstance(self.agent, StateMachineAgent)
        
        # Title
        if is_limited_perception:
            title = self.font.render("Agent's Limited Observation (TF, ML, MR only)", True, (255, 255, 255))
        else:
            title = self.font.render("Agent's 3x3 Observation", True, (255, 255, 255))
        self.screen.blit(title, (x, y - 30))
        
        # Grid labels (from agent's perspective)
        # TF = Top Front (what's in front)
        # ML = Middle Left, MR = Middle Right
        # AG = Agent at center
        labels = [
            ['TL', 'TF', 'TR'],
            ['ML', 'AG', 'MR'],
            ['BL', 'BF', 'BR']
        ]
        
        cell_size = 60
        from environments.maze_env import Coin, Wall
        
        for row in range(3):
            for col in range(3):
                # Skip cells that StateMachineAgent cannot see
                if is_limited_perception:
                    # Only show TF (1,0), ML (0,1), AG (1,1), MR (2,1)
                    if not ((row == 0 and col == 1) or  # TF
                           (row == 1 and col == 0) or  # ML
                           (row == 1 and col == 1) or  # AG
                           (row == 1 and col == 2)):   # MR
                        continue
                
                cell_x = x + col * cell_size
                cell_y = y + row * cell_size
                
                # Get cell from the observation grid
                cell = obs_grid.get(col, row)
                
                # Determine color and text based on cell type
                if row == 1 and col == 1:
                    # Agent position - always show facing up since view is rotated
                    color = (100, 200, 100)  # Agent (green)
                    label_text = '^'
                elif isinstance(cell, Wall):
                    color = (80, 80, 80)  # Wall (gray)
                    label_text = 'W'
                elif isinstance(cell, Coin):
                    color = (255, 215, 0)  # Reward (gold)
                    label_text = 'C'
                elif cell is None:
                    color = (200, 200, 200)  # Empty (light gray)
                    label_text = ''
                else:
                    # Unknown object
                    color = (150, 150, 150)
                    label_text = '?'
                
                # Draw cell
                pygame.draw.rect(self.screen, color, (cell_x, cell_y, cell_size - 2, cell_size - 2))
                pygame.draw.rect(self.screen, (255, 255, 255), (cell_x, cell_y, cell_size - 2, cell_size - 2), 1)
                
                # Draw label (position label at top)
                pos_label = self.font_small.render(labels[row][col], True, (0, 0, 0))
                self.screen.blit(pos_label, (cell_x + 5, cell_y + 5))
                
                # Draw content (larger, centered)
                if label_text:
                    content_font = pygame.font.Font(None, 48)
                    content_label = content_font.render(label_text, True, (0, 0, 0) if row != 1 or col != 1 else (255, 255, 255))
                    content_rect = content_label.get_rect(center=(cell_x + cell_size // 2, cell_y + cell_size // 2))
                    self.screen.blit(content_label, content_rect)
    
    def _render_step_info(self, step_data, x, y):
        """Render step information and agent decision"""
        # Handle initial state display
        if step_data['action'] is None:
            step_display = "Initial State"
        else:
            step_display = f"Step: {step_data['step'] + 1} / {self.total_steps - 1}"
        
        info_lines = [
            step_display,
            f"Position: {step_data['agent_pos']}",
            f"Direction: {['>', 'v', '<', '^'][step_data['agent_dir']]}",
            f"",
        ]
        
        # Only show action if one was taken
        if step_data['action'] is not None:
            info_lines.extend([
                f"Action: {['Turn Left', 'Turn Right', 'Move Forward'][step_data['action']]}",
                f"Reward: {step_data['reward']:.1f}",
                f"Total Reward: {step_data['total_reward']:.1f}",
            ])
        else:
            info_lines.extend([
                "Action: (waiting for first action)",
                f"Reward: 0.0",
                f"Total Reward: 0.0",
            ])
        
        # Add agent decision explanation if available
        if hasattr(self.agent, 'get_decision_explanation'):
            try:
                explanation = self.agent.get_decision_explanation()
                info_lines.append("")
                info_lines.append("Decision Reasoning:")
                info_lines.extend(explanation.split('\n'))
            except:
                pass
        
        for i, line in enumerate(info_lines):
            text = self.font_small.render(line, True, (255, 255, 255))
            self.screen.blit(text, (x, y + i * 25))
    
    def _render_controls(self):
        """Render control instructions"""
        y = self.window_height - 120
        
        controls = [
            "Controls:",
            "<- /->: Previous/Next step  |  Space: Play/Pause  |  R: Reset",
            f"1-9: Set speed ({self.playback_speed} steps/sec)  |  Q/ESC: Quit",
            f"Status: {'PLAYING' if self.playing else 'PAUSED'}"
        ]
        
        for i, line in enumerate(controls):
            text = self.font_small.render(line, True, (200, 200, 200))
            self.screen.blit(text, (20, y + i * 25))


def main():
    """Main function to run the visualizer"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Step-by-step agent visualization')
    parser.add_argument('--env', type=str, default='simple', 
                       choices=['simple', 'medium', 'hard'],
                       help='Environment to use (default: simple)')
    parser.add_argument('--agent', type=str, default='naive',
                       choices=['naive', 'production_rules', 'state_machine'],
                       help='Agent type (default: naive)')
    parser.add_argument('--max-steps', type=int, default=1000,
                       help='Maximum steps to record (default: 1000)')
    
    args = parser.parse_args()
    
    # Create environment with custom start positions for each map
    # Format: {env_name: (position, direction)}
    # Direction: 0=right, 1=down, 2=left, 3=up
    start_configs = {
        'simple': {'agent_start_pos': (2, 2), 'agent_start_dir': 3},
        'medium': {'agent_start_pos': (3, 10), 'agent_start_dir': 0},
        'hard': {'agent_start_pos': (12, 8), 'agent_start_dir': 2},
    }
    
    env_map = {
        'simple': SimpleMazeEnv,
        'medium': MediumMazeEnv,
        'hard': HardMazeEnv
    }
    
    # Create environment with custom starting position
    env = env_map[args.env](**start_configs[args.env])
    
    # Create agent
    if args.agent == 'naive':
        agent = NaiveAgent(env.action_space)
        print("Using Naive Agent (Random Movement)")
    elif args.agent == 'production_rules':
        agent = ProductionRulesAgent(env.action_space)
        print("Using Production Rules Agent (Left-hand following)")
    elif args.agent == 'state_machine':
        agent = StateMachineAgent(env.action_space)
        print("Using State Machine Agent (Left-hand wall following with limited perception)")
    
    # Create visualizer and run
    visualizer = StepVisualizer(env, agent, max_steps=args.max_steps)
    visualizer.run()


if __name__ == '__main__':
    main()
