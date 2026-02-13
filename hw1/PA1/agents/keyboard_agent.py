"""
Keyboard Agent - Manual control for testing and human baseline
"""

from agents.base_agent import BaseAgent

class KeyboardAgent(BaseAgent):
    """Agent controlled by keyboard input for manual testing"""
    
    def __init__(self, action_space):
        """
        Initialize the keyboard agent
        
        Args:
            action_space: The action space from the environment
        """
        super().__init__(action_space)
        self.next_action = None
        
    def perceive(self, observation):
        """
        Process observation (keyboard agent doesn't need special processing)
        
        Args:
            observation: Dictionary containing 'image', 'direction', and 'mission'
        
        Returns:
            observation: Raw observation
        """
        return observation
    
    def set_action(self, action):
        """
        Set the next action from keyboard input
        
        Args:
            action: Action index (0=turn left, 1=turn right, 2=move forward)
        """
        self.next_action = action
    
    def decide(self):
        """
        Return the action set by keyboard input
        
        Returns:
            int: Action index or None if no action set
        """
        action = self.next_action
        self.next_action = None  # Clear after returning
        return action
    
    def reset(self):
        """Reset agent state for a new episode"""
        super().reset()
        self.next_action = None
