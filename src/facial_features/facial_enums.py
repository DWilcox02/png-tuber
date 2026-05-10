from enum import Enum

class EyeStatus(Enum):
    OPEN = "Eye Open"
    WIDE = "Eye Wide"
    CLOSED = "Eye Closed"

class MouthStatus(Enum):
    OPEN = "Mouth Open"
    CLOSED = "Mouth Closed"

class SmileStatus(Enum):
    SMILING = "Smiling"
    NEUTRAL = "Neutral"
    FROWNING = "Frowning"