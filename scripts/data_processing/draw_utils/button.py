class ArrowButton:

    def __init__(self, button_id, circle, direction, step_size=5):
        self.id = button_id
        self.circle = circle
        self.direction = direction  # 'up' or 'down'
        self.step_size = step_size

    def contains(self, event):
        return self.circle.contains(event)[0]

    def get_offset(self):
        """Get the y-offset based on button direction"""
        return -self.step_size if self.direction == 'up' else self.step_size
