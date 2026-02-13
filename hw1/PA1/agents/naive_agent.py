"""
Naive Agent - Random movement baseline
"""

import random
from agents.base_agent import BaseAgent


class NaiveAgent(BaseAgent):
    """
    Naive agent that makes random decisions.
    This serves as a baseline for comparison.
    """
    
    def __init__(self, action_space):
        """
        Initialize the naive agent
        
        Args:
            action_space: The action space from the environment
        """
        super().__init__(action_space)
        self.last_observation = None
        
    def perceive(self, observation):
        """
        Process observation (naive agent just stores it)
        
        Args:
            observation: Dictionary containing 'image', 'direction', and 'mission'
        
        Returns:
            observation: Raw observation
        """
        self.last_observation = observation
        return observation
    
    def decide(self):
        """
        Make a random decision
        
        Returns:
            int: Random action index (0=turn left, 1=turn right, 2=move forward)
        """
        action = random.randint(0, 2)
        return action
    
    def reset(self):
        """Reset agent state for a new episode"""
        super().reset()
        self.last_observation = None
