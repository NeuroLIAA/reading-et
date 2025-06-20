from .circle import FixCircle
from .line import HLine
from . import drawing


def advance_sequence(event, state, screens, screens_sequence, sequence_states, ax, fig, editable):
    prev_seq = state['sequence_index']
    if event.key == 'right' and prev_seq < len(screens_sequence) - 1:
        state['sequence_index'] += 1
    elif event.key == 'left' and prev_seq > 0:
        state['sequence_index'] -= 1
    current_seqid = state['sequence_index']
    if prev_seq != current_seqid:
        for cid in state['cids']:
            fig.canvas.mpl_disconnect(cid)
        drawing.update_figure(state, fig, ax, screens, sequence_states, editable)


def onclick(event, circles, arrows, fig, ax, last_actions, df_fix, lines_coords, hlines, buttons=None):
    if event.button == 1:
        # Check if a button was clicked first
        if buttons and handle_button_click(event, buttons, hlines, lines_coords, fig):
            return
        handle_click(event, hlines, circles, last_actions)
    elif event.button == 2:
        remove_fixation(event, circles, arrows, ax, last_actions, df_fix)
    elif event.button == 3:
        undo_lastaction(last_actions, circles, arrows, ax, lines_coords, df_fix)
    fig.canvas.draw()


def handle_button_click(event, buttons, hlines, lines_coords, fig):
    """Handle clicks on arrow buttons"""
    for button in buttons:
        if button.contains(event):
            offset = button.get_offset()
            drawing.move_horizontal_lines(hlines, lines_coords, offset)
            # Add visual feedback by briefly changing button appearance
            original_alpha = button.circle.get_alpha()
            button.circle.set_alpha(1.0)
            fig.canvas.draw()
            # Reset alpha after a brief moment (this happens immediately in practice)
            button.circle.set_alpha(original_alpha)
            return True
    return False


def handle_click(event, hlines, circles, last_actions):
    clicked_fixation = select_fixation(event, circles, last_actions)
    if not clicked_fixation:
        select_hline(event, hlines, last_actions)


def release_object(event, lines_coords, df_fix, last_actions):
    if event.button == 1 and last_actions:
        selected_object = last_actions[-1]
        if selected_object.is_selected:
            if isinstance(selected_object, FixCircle):
                selected_object.deselect(df_fix)
            else:
                selected_object.deselect(lines_coords)


def update_arrows(ax, arrows, circles, index, remove_current=True):
    if index > 0:
        if remove_current:
            arrows[index - 1].remove(), arrows.pop(index - 1)
        new_arrow = drawing.draw_arrow(ax, circles[index - 1].center(), circles[index].center(),
                                       circles[index - 1].color())
        arrows.insert(index - 1, new_arrow)
    if index < len(circles) - 1:
        if remove_current:
            arrows[index].remove(), arrows.pop(index)
        new_arrow = drawing.draw_arrow(ax, circles[index].center(), circles[index + 1].center(),
                                       circles[index].color())
        arrows.insert(index, new_arrow)


def move_object(event, ax, arrows, circles, last_actions):
    if not last_actions:
        return
    selected_object = last_actions[-1]
    if selected_object.is_selected:
        selected_object.update_coords(event.xdata, event.ydata)
        if isinstance(selected_object, FixCircle):
            update_arrows(ax, arrows, circles, selected_object.id)
        selected_object.draw_canvas()


def select_hline(event, hlines, last_actions):
    for line in hlines:
        if line.contains(event):
            line.select()
            last_actions.append(line)
            break


def select_fixation(event, circles, last_actions):
    for circle in circles:
        if circle.contains(event):
            circle.select()
            last_actions.append(circle)
            return circle


def undo_lastaction(last_actions, circles, arrows, ax, lines_coords, df_fix):
    if last_actions:
        last_action = last_actions.pop()
        if isinstance(last_action, FixCircle) and last_action.was_removed:
            fix_circle = last_action
            index = fix_circle.id
            update_indices(circles, index, 1)
            fix_circle.restore(circles, ax, df_fix)

            if 0 < index < len(circles) - 1:
                arrows[index - 1].remove(), arrows.pop(index - 1)
            update_arrows(ax, arrows, circles, index, remove_current=False)
        elif isinstance(last_action, HLine) and not last_action.is_selected:
            line = last_action
            line.restore_y()
            lines_coords[line.id] = line.get_y()


def remove_fixation(event, circles, arrows, ax, last_actions, df_fix):
    for i, fix_circle in enumerate(circles):
        if fix_circle.contains(event):
            if i < len(circles) - 1:
                arrows[i].remove(), arrows.pop(i)
            if i > 0:
                arrows[i - 1].remove(), arrows.pop(i - 1)
            if 0 < i < len(circles) - 1:
                new_arrow = drawing.draw_arrow(ax, circles[i - 1].center(), circles[i + 1].center(), fix_circle.color())
                arrows.insert(i - 1, new_arrow)
            last_actions.append(fix_circle)
            fix_circle.remove(circles, df_fix)
            update_indices(circles, fix_circle.id, -1)
            break


def update_indices(circles, from_index, value):
    for circle in circles[from_index:]:
        circle.id += value