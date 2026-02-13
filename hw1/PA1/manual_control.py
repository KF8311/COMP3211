"""
Manual control script for testing the maze environment with keyboard input
"""

import sys
import argparse
import gymnasium as gym
from environments.maze_env import SimpleMazeEnv, MediumMazeEnv, HardMazeEnv
from agents import KeyboardAgent


def main():
    parser = argparse.ArgumentParser(description='Manual control for maze navigation')
    parser.add_argument(
        '--env',
        type=str,
        default='simple',
        choices=['simple', 'medium', 'hard'],
        help='Maze difficulty (simple, medium, hard)'
    )
    parser.add_argument(
        '--view',
        type=str,
        default='agent',
        choices=['agent', 'full'],
        help='View mode: agent (3x3 observation) or full (complete map)'
    )
    parser.add_argument(
        '--tile-size',
        type=int,
        default=32,
        help='Size of tiles in pixels'
    )
    
    args = parser.parse_args()
    
    # Create environment
    if args.env == 'simple':
        env = SimpleMazeEnv(
            render_mode='human', 
            tile_size=args.tile_size,
            agent_start_pos=(2, 2),  # Change position here
            agent_start_dir=3  # Change direction here (0=right, 1=down, 2=left, 3=up)
        )
    elif args.env == 'medium':
        env = MediumMazeEnv(
            render_mode='human', 
            tile_size=args.tile_size,
            agent_start_pos=(3, 10),  # Change position here
            agent_start_dir=0  # Change direction here
        )
    else:
        env = HardMazeEnv(
            render_mode='human', 
            tile_size=args.tile_size,
            agent_start_pos=(12, 8),  # Change position here
            agent_start_dir=2  # Change direction here
        )
    
    # Enable agent POV for centered view when in agent view mode
    if args.view == 'agent':
        env.agent_pov = True
    else:
        env.agent_pov = False
    
    # Create keyboard agent
    agent = KeyboardAgent(env.action_space)
    
    print("\n" + "="*60)
    print("COMP3211 Assignment 1 - Manual Control")
    print("="*60)
    print(f"Environment: {args.env.capitalize()} Maze")
    print(f"View Mode: {args.view.capitalize()}")
    print(f"Max Steps: {env.max_steps}")
    print("\nControls:")
    print("  Arrow Up / W    : Move forward")
    print("  Arrow Left / A  : Turn left")
    print("  Arrow Right / D : Turn right")
    print("  R               : Reset environment")
    print("  Q / ESC         : Quit")
    print("="*60 + "\n")
    
    # Reset environment
    obs, info = env.reset()
    agent.reset()
    
    # Render initial state
    env.render()
    
    print(f"Mission: {env.mission}")
    print(f"Total Rewards: {env.total_rewards}")
    print(f"Agent View Size: {env.agent_view_size}x{env.agent_view_size}")
    print("\nStarting manual control... Use arrow keys to move.\n")
    
    # Manual control loop
    try:
        manual_control_loop(env, agent, args.view)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        env.close()
        print("Environment closed.")


def manual_control_loop(env, agent, view_mode):
    """
    Main control loop for manual navigation
    
    Args:
        env: The maze environment
        agent: The keyboard agent
        view_mode: 'agent' or 'full' view
    """
    import pygame
    
    # Initialize pygame for keyboard input
    pygame.init()
    
    terminated = False
    truncated = False
    total_reward = 0
    
    while True:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            
            if event.type == pygame.KEYDOWN:
                action = None
                
                # Map keys to actions
                if event.key in [pygame.K_UP, pygame.K_w]:
                    action = 2  # Move forward
                elif event.key in [pygame.K_LEFT, pygame.K_a]:
                    action = 0  # Turn left
                elif event.key in [pygame.K_RIGHT, pygame.K_d]:
                    action = 1  # Turn right
                elif event.key == pygame.K_r:
                    # Reset
                    print("\n" + "="*60)
                    print("RESETTING ENVIRONMENT")
                    print("="*60)
                    obs, info = env.reset()
                    agent.reset()
                    terminated = False
                    truncated = False
                    total_reward = 0
                    env.render()
                    print_status(agent, env, total_reward)
                    continue
                elif event.key in [pygame.K_q, pygame.K_ESCAPE]:
                    return
                
                # Execute action if valid
                if action is not None and not (terminated or truncated):
                    obs, reward, terminated, truncated, info = env.step(action)
                    agent.update_stats(reward, terminated or truncated)
                    total_reward += reward
                    
                    # Render
                    env.render()
                    
                    # Print status
                    print_status(agent, env, total_reward, reward, info)
                    
                    # Check if episode ended
                    if terminated:
                        print("\n" + "="*60)
                        print("EPISODE COMPLETED!")
                        if info.get('success', False):
                            print("SUCCESS! All rewards collected!")
                        print(f"Final Score: {agent.rewards_collected}/{env.total_rewards} rewards")
                        print(f"Total Steps: {agent.steps_taken}")
                        print(f"Efficiency: {agent.rewards_collected/agent.steps_taken:.3f} rewards/step")
                        print("="*60)
                        print("Press R to reset or Q to quit")
                    elif truncated:
                        print("\n" + "="*60)
                        print("EPISODE TRUNCATED (Max steps reached)")
                        print(f"Final Score: {agent.rewards_collected}/{env.total_rewards} rewards")
                        print(f"Total Steps: {agent.steps_taken}")
                        print("="*60)
                        print("Press R to reset or Q to quit")
        
        # Small delay to prevent high CPU usage
        pygame.time.wait(10)


def print_status(agent, env, total_reward, last_reward=0, info=None):
    """Print current status information"""
    status = f"Steps: {agent.steps_taken:3d} | "
    status += f"Rewards: {agent.rewards_collected:2d}/{env.total_rewards:2d} | "
    status += f"Total Reward: {total_reward:.1f}"
    
    if info and info.get('reward_collected', False):
        status += " | +1 REWARD COLLECTED!"
    
    print(status)


if __name__ == '__main__':
    main()
