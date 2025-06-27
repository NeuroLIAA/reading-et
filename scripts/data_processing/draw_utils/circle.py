class FixCircle:
    def __init__(self, id, circle, ann, fixation):
        self.id = id
        self.circle = circle
        self.ann = ann
        self.fixation = fixation
        self.is_selected = False
        self.was_removed = False
    
    def center(self):
        return self.circle.center
    
    def contains(self, event):
        return self.circle.contains(event)[0]
    
    def fix_name(self):
        return self.fixation.name

    def select(self):
        self.is_selected = True

    def deselect(self, df_fix):
        self.is_selected = False
        df_fix.loc[self.fix_name(), ['xAvg', 'yAvg']] = self.center()

    def update_coords(self, x, y):
        self.circle.center = self.circle.center[0], y
        self.ann.set_position((self.circle.center[0], y))

    def remove(self, circles, df_fix):
        circles.pop(self.id)
        df_fix.drop(self.fix_name(), inplace=True)
        self.circle.remove()
        self.ann.remove()
        self.was_removed = True
    
    def add_to_axes(self, ax):
        ax.add_patch(self.circle), ax.add_artist(self.ann)
        self.was_removed = False

    def draw_canvas(self):
        self.circle.figure.canvas.draw()

    def restore(self, circles, ax, df_fix):
        circles.insert(self.id, self)
        self.add_to_axes(ax)
        df_fix.loc[self.fix_name()] = self.fixation
        df_fix.sort_index(inplace=True)

    def color(self):
        return self.circle.get_facecolor()
