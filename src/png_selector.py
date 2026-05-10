from facial_features.facial_enums import EyeStatus, MouthStatus

class PngSelector():
    """Handles the selection of PNG assets based on facial state."""
    
    def __init__(self):
        # Map facial states to asset paths
        self.assets = {
            "default": "assets/larry.jpeg",
            "tongue": "assets/cat-tongue.jpeg",
            "shock": "assets/cat-shock.jpeg",
            "glare": "assets/cat-glare.jpeg",
        }

    def select_image(self, status):
        """
        Determines the appropriate cat image based on the provided FacialStatus.
        """
        cat_image = self.assets["default"]

        # Expression priority logic
        if status.mouth == MouthStatus.OPEN:
            cat_image = self.assets["tongue"]
        elif status.left_eye == EyeStatus.WIDE or status.right_eye == EyeStatus.WIDE:
            cat_image = self.assets["shock"]
        elif status.left_eye == EyeStatus.CLOSED or status.right_eye == EyeStatus.CLOSED:
            cat_image = self.assets["glare"]

        return cat_image

    def terminate(self):
        pass
