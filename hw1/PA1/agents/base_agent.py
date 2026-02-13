"""
Base Agent class for all agents
"""

class BaseAgent:
    """Base class for all agents in the maze navigation task"""
    
    def __init__(self, action_space):
        """
        Initialize the agent
        
        Args:
            action_space: The action space from the environment
        """
        self.action_space = action_space
        self.rewards_collected = 0
        self.steps_taken = 0
        
    def perceive(self, observation):
        """
        Process the observation from the environment
        
        Args:
            observation: Dictionary containing 'image', 'direction', and 'mission'
        
        Returns:
            Processed observation (implementation specific)
        """
        raise NotImplementedError("Subclasses must implement perceive()")
    
    def decide(self):
        """
        Decide on the next action to take
        
        Returns:
            int: Action index (0=turn left, 1=turn right, 2=move forward)
        """
        raise NotImplementedError("Subclasses must implement decide()")
    
    def reset(self):
        """Reset agent state for a new episode"""
        self.rewards_collected = 0
        self.steps_taken = 0
    
    def update_stats(self, reward, done):
        """
        Update agent statistics after taking an action
        
        Args:
            reward: Reward received
            done: Whether episode is finished
        """
        if reward > 0:
            self.rewards_collected += 1
        self.steps_taken += 1
