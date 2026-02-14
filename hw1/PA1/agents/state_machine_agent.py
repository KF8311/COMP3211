"""
State Machine Agent - Wall-following with limited perception

This agent uses a state machine with three states to follow walls with
limited sensory input (can only see ML, TF, MR).
"""

from agents.base_agent import BaseAgent
from minigrid.core.constants import OBJECT_TO_IDX


class StateMachineAgent(BaseAgent):
    """
    State machine agent for wall-following with limited perception.

    Perception Constraint: Can only see ML, TF, MR (other sensors impaired)

    States:
    - FIND_WALL: Initial state. Agent searches for a left wall to follow.
    - FOLLOW_WALL: Operating state when wall on left is present. Moving forward while maintaining wall contact.
    - TURN_CORNER: Activated when wall ahead and on left. Manages multi-step turn.
    """

    # State definitions
    FOLLOW_WALL = "FOLLOW_WALL"
    FIND_WALL = "FIND_WALL"
    TURN_CORNER = "TURN_CORNER"

    def __init__(self, action_space):
        """
        Initialize the state machine agent

        Args:
            action_space: The action space from the environment
        """
        super().__init__(action_space)
        self.state = self.FIND_WALL
        self.last_observation = None

        # Action constants
        self.TURN_LEFT = 0
        self.TURN_RIGHT = 1
        self.MOVE_FORWARD = 2

    def perceive(self, observation):
        """
        Process observation - LIMITED to ML, TF, MR only

        Args:
            observation: Dictionary containing 'image', 'direction', and 'mission'

        Returns:
            dict: Processed perception with boolean flags for walls
        """
        self.last_observation = observation
        image = observation["image"]

        # Limited perception: can only see ML, TF, MR
        # Grid layout (agent's perspective):
        # [XX] [TF] [XX]    (1, 0)
        # [ML] [AG] [MR]    (0, 1) (1, 1) (2, 1)
        # [XX] [XX] [XX]

        wall_idx = OBJECT_TO_IDX["wall"]

        perception = {
            "TF": image[1, 0, 0] == wall_idx,  # Top-Front
            "ML": image[0, 1, 0] == wall_idx,  # Mid-Left
            "MR": image[2, 1, 0] == wall_idx,  # Mid-Right
        }

        return perception

    def decide(self):
        """
        Make decision based on current state and perception

        Returns:
            int: Action index (0=turn left, 1=turn right, 2=move forward)
        """
        if self.last_observation is None:
            return self.MOVE_FORWARD

        p = self.perceive(self.last_observation)
        return self._decide_left_hand(p)

    def _decide_left_hand(self, p):
        """
        State machine logic for left-hand wall following

        Args:
            p: Perception dictionary with boolean flags for walls

        Returns:
            int: Action to take
        """
        # TODO: Implement state machine logic for left-hand wall following
        #
        # Extract wall information from perception
        wall_ahead = p["TF"]
        wall_on_left = p["ML"]
        wall_on_right = p["MR"]

        # TODO: Implement behavior for FOLLOW_WALL state
        # Conditions to check:
        # - If wall ahead: transition to TURN_CORNER and turn right
        # - If no wall on left: transition to FIND_WALL and turn left
        # - Otherwise: stay in FOLLOW_WALL and move forward
        if self.state == self.FOLLOW_WALL:
            if wall_ahead:
                self.state = self.TURN_CORNER
                return self.TURN_RIGHT
            if wall_on_left == False:
                self.state = self.FIND_WALL
                return self.TURN_LEFT
            else:
                return self.MOVE_FORWARD

        # TODO: Implement behavior for FIND_WALL state
        # Conditions to check (in order):
        # - If wall ahead: transition to FOLLOW_WALL and turn right
        # - If wall on left: transition to FOLLOW_WALL and move forward
        # - If wall on right: transition to TURN_CORNER and turn right
        # - Otherwise: stay in FIND_WALL and move forward
        elif self.state == self.FIND_WALL:
            if wall_ahead:
                self.state = self.FOLLOW_WALL
                return self.TURN_RIGHT
            if wall_on_left:
                self.state = self.FOLLOW_WALL
                return self.MOVE_FORWARD
            else:
                return self.MOVE_FORWARD

        # TODO: Implement behavior for TURN_CORNER state
        # Conditions to check:
        # - If wall on left AND no wall ahead: transition to FOLLOW_WALL and move forward
        # - Otherwise: stay in TURN_CORNER and keep turning right
        elif self.state == self.TURN_CORNER:
            if wall_on_left and wall_ahead == False:
                self.state = self.FOLLOW_WALL
                return self.MOVE_FORWARD
            else:
                return self.TURN_RIGHT

        # Default fallback (should not reach here if all states are implemented correctly)
        return self.MOVE_FORWARD

    def reset(self):
        """Reset agent state for a new episode"""
        super().reset()
        self.state = self.FIND_WALL
        self.last_observation = None
