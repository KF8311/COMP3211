"""
Production Rules Agent - Boundary Following Agent using Left-Hand Rule

This agent follows the left-hand wall-following strategy using production rules.
"""

from agents.base_agent import BaseAgent
from minigrid.core.constants import OBJECT_TO_IDX


class ProductionRulesAgent(BaseAgent):
    """
    Boundary Following Agent using Production Rules (Left-Hand Rule)
    
    The agent follows walls using the left-hand rule to systematically
    explore the maze and collect rewards along boundaries.

    3x3 agent observation grid (agent is at center [1, 1])
    Grid layout (agent's perspective, facing "up" in view):
    [TL] [TF] [TR]    TL=Top-Left,  TF=Top-Front, TR=Top-Right
    [ML] [AG] [MR]    ML=Mid-Left,  AG=Agent,     MR=Mid-Right 
    [BL] [BF] [BR]    BL=Bot-Left,  BF=Bot-Front, BR=Bot-Right

    
    Production Rules for Left-Hand Following (in priority order):
    1. IF (TF == WALL OR TR == WALL) THEN TURN_RIGHT
       Rationale: Wall ahead or ahead-right blocks path—must turn away.
    2. ELSE IF (ML == EMPTY AND BL == WALL) THEN TURN_LEFT
       Rationale: Left side partially open (ML empty, BL has wall)—turn left to maintain wall contact.
    3. ELSE MOVE_FORWARD
       Rationale: Default action when no obstacles ahead and left wall maintained.
    """
    
    def __init__(self, action_space):
        """
        Initialize the Production Rules Agent
        
        Args:
            action_space: The action space from the environment
        """
        super().__init__(action_space)
        self.last_observation = None
        
        # Action constants
        self.TURN_LEFT = 0
        self.TURN_RIGHT = 1
        self.MOVE_FORWARD = 2
        
    def perceive(self, observation):
        """
        Process the 3x3 observation window
        
        Args:
            observation: Dictionary containing 'image', 'direction', and 'mission'
                        image is 3x3x3 array (width, height, channels)
        
        Returns:
            dict: Processed observation with boolean flags for walls in each position
        """
        self.last_observation = observation
        image = observation['image']
        
        # Extract the 3×3 grid (agent is at center [1, 1])
        # Grid layout (agent's perspective, facing "up" in view):
        # [TL] [TF] [TR]    (0, 0) (1, 0) (2, 0)
        # [ML] [AG] [MR]    (0, 1) (1, 1) (2, 1)
        # [BL] [BF] [BR]    (0, 2) (1, 2) (2, 2)
        
        wall_idx = OBJECT_TO_IDX['wall']
        
        # Check each position for walls
        perception = {
            'TL': image[0, 0, 0] == wall_idx,  # Top-Left
            'TF': image[1, 0, 0] == wall_idx,  # Top-Front
            'TR': image[2, 0, 0] == wall_idx,  # Top-Right
            'ML': image[0, 1, 0] == wall_idx,  # Mid-Left
            'MR': image[2, 1, 0] == wall_idx,  # Mid-Right
            'BL': image[0, 2, 0] == wall_idx,  # Bot-Left
            'BF': image[1, 2, 0] == wall_idx,  # Bot-Front
            'BR': image[2, 2, 0] == wall_idx,  # Bot-Right
        }
        
        return perception
    
    def decide(self):
        """
        Decide next action using production rules
        
        Returns:
            int: Action index (0=turn left, 1=turn right, 2=move forward)
        """
        if self.last_observation is None:
            return self.MOVE_FORWARD
        
        # Get perception of walls
        p = self.perceive(self.last_observation)
        return self._decide_left_hand(p)
    
    def _decide_left_hand(self, p):
        """
        Apply left-hand following production rules
        
        Args:
            p: Perception dictionary with boolean flags for walls
        
        Returns:
            int: Action to take
        """
        # TODO: Implement the three production rules in priority order
        # 
        # Rule 1: IF (TF == WALL OR TR == WALL) THEN TURN_RIGHT
        #         Rationale: Wall ahead or ahead-right blocks path—must turn away.
        # 
        # Rule 2: ELSE IF (ML == EMPTY AND BL == WALL) THEN TURN_LEFT
        #         Rationale: Left side partially open (ML empty, BL has wall)—turn left to maintain wall contact.
        # 
        # Rule 3: ELSE MOVE_FORWARD
        #         Rationale: Default action when no obstacles ahead and left wall maintained.
        
        # Hint: Read the above perceive() method to understand how to use `p`
        # Hint: Return one of: self.TURN_LEFT, self.TURN_RIGHT, self.MOVE_FORWARD
        
        pass  # Replace this with your implementation
    
    def reset(self):
        """Reset agent state for a new episode"""
        super().reset()
        self.last_observation = None
