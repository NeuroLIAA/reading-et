import numpy as np


class ArrowButton:
    """Button class similar to FixCircle for handling arrow buttons"""

    def __init__(self, button_id, circle, annotation, direction, step_size=5):
        self.id = button_id
        self.circle = circle
        self.annotation = annotation
        self.direction = direction  # 'up' or 'down'
        self.step_size = step_size
        self.is_selected = False

    def contains(self, event):
        """Check if click event is within the button circle"""
        if event.xdata is None or event.ydata is None:
            return False
        center_x, center_y = self.circle.center
        distance = np.sqrt((event.xdata - center_x) ** 2 + (event.ydata - center_y) ** 2)
        return distance <= self.circle.radius

    def get_offset(self):
        """Get the y-offset based on button direction"""
        return -self.step_size if self.direction == 'up' else self.step_size
