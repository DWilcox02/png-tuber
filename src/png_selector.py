from facial_features.facial_enums import EyeStatus, MouthStatus


class PngSelector:
    """Handles the selection of PNG assets based on facial state."""

    def __init__(self):
        # The default asset if no rules match
        self.default_asset = "assets/eyes_open_mouth_closed.png"

        # A scalable list of rules. Order determines priority (first match wins).
        # If an asset only cares about one feature (e.g., just the mouth),
        # you can simply omit the other features from the "conditions" dict!
        self.asset_rules = [
            {
                "path": "assets/eyes_closed_mouth_open.png",
                "conditions": {"eyes": EyeStatus.CLOSED, "mouth": MouthStatus.OPEN},
            },
            {
                "path": "assets/eyes_closed_mouth_closed.png",
                "conditions": {"eyes": EyeStatus.CLOSED, "mouth": MouthStatus.CLOSED},
            },
            {
                "path": "assets/eyes_open_mouth_open.png",
                "conditions": {"eyes": EyeStatus.OPEN, "mouth": MouthStatus.OPEN},
            },
            {
                "path": "assets/eyes_open_mouth_closed.png",
                "conditions": {"eyes": EyeStatus.OPEN, "mouth": MouthStatus.CLOSED},
            },
        ]

    def _get_unified_eye_state(self, status):
        """
        Helper to collapse left/right eye status into a single unified state.
        Currently returns CLOSED if *either* eye is closed (e.g. winking).
        Treats anything else (OPEN, WIDE) as simply OPEN.
        """
        if status.left_eye == EyeStatus.CLOSED or status.right_eye == EyeStatus.CLOSED:
            return EyeStatus.CLOSED
        return EyeStatus.OPEN

    def select_image(self, status):
        """
        Determines the appropriate cat image by evaluating the rule list against
        the current facial status.
        """
        # 1. Flatten the current status into a dictionary of active features
        active_features = {
            "mouth": status.mouth,
            "eyes": self._get_unified_eye_state(status),
            # As you expand the app, easily add future parameters here:
            # "smile": status.smile,
            # "is_talking": status.is_talking
        }

        # 2. Iterate through rules and return the path of the first full match
        for rule in self.asset_rules:
            conditions = rule["conditions"]

            # Check if every requirement in this rule is currently true
            if all(active_features.get(key) == value for key, value in conditions.items()):
                return rule["path"]

        return self.default_asset

    def terminate(self):
        pass
