from facial_features.facial_enums import EyeStatus, MouthStatus


class PngSelector:
    def __init__(self):
        self.default_asset = "assets/eyes_open_mouth_closed.png"

        # Expandable list of rules. First matching rule wins.
        self.asset_rules = [
            # -- WIDE EYES --
            {
                "path": "assets/eyes_wide_mouth_open.png",
                "conditions": {"eyes": EyeStatus.WIDE, "mouth": MouthStatus.OPEN},
            },
            {
                "path": "assets/eyes_wide_mouth_closed.png",
                "conditions": {"eyes": EyeStatus.WIDE, "mouth": MouthStatus.CLOSED},
            },
            # -- CLOSED EYES --
            {
                "path": "assets/eyes_closed_mouth_open.png",
                "conditions": {"eyes": EyeStatus.CLOSED, "mouth": MouthStatus.OPEN},
            },
            {
                "path": "assets/eyes_closed_mouth_closed.png",
                "conditions": {"eyes": EyeStatus.CLOSED, "mouth": MouthStatus.CLOSED},
            },
            # -- OPEN EYES --
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
        Collapses left/right eye status into a single unified state.
        Priority Hierarchy: CLOSED > WIDE > OPEN
        """
        # 1. Winking or blinking
        if status.left_eye == EyeStatus.CLOSED or status.right_eye == EyeStatus.CLOSED:
            return EyeStatus.CLOSED

        # 2. Looking shocked/surprised
        if status.left_eye == EyeStatus.WIDE or status.right_eye == EyeStatus.WIDE:
            return EyeStatus.WIDE

        # 3. Default resting state
        return EyeStatus.OPEN

    def select_image(self, status):
        # Flatten current state for the evaluation engine
        active_features = {
            "mouth": status.mouth,
            "eyes": self._get_unified_eye_state(status),
        }

        # Find the first image where all required conditions are met
        for rule in self.asset_rules:
            conditions = rule["conditions"]
            if all(active_features.get(key) == value for key, value in conditions.items()):
                return rule["path"]

        return self.default_asset

    def terminate(self):
        pass
