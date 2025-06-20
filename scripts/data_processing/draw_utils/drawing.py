from scripts.data_processing.draw_utils.button import ArrowButton
from scripts.data_processing.draw_utils.circle import FixCircle
from scripts.data_processing.draw_utils.line import HLine
from scripts.data_processing.draw_utils.handles import onclick, move_object, release_object
from PIL import Image, ImageDraw
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def update_figure(state, fig, ax, screens, sequence_states, editable):
    current_seqid = state['sequence_index']
    screenid, fixations, lines = sequence_states[current_seqid]['screenid'], \
        sequence_states[current_seqid]['fixations'], \
        sequence_states[current_seqid]['lines']
    state['cids'] = draw_scanpath(screens[screenid], fixations, fig, ax,
                                  title=f'Screen {screenid}',
                                  lines_coords=lines,
                                  editable=editable)


def draw_scanpath(img, df_fix, fig, ax, ann_size=8, fix_size=15, min_t=250, title=None, lines_coords=None,
                  editable=False):
    """ df_fix: pd.DataFrame with columns: ['xAvg', 'yAvg', 'duration'] """
    """ Given a scanpath, draw on the img using the fig and axes """
    """ The duration of each fixation is used to determine the size of each circle """
    ax.clear()
    ax.imshow(img, cmap=mpl.colormaps['gray'])
    if title:
        ax.set_title(title)

    xs, ys, ts = df_fix['xAvg'].to_numpy(dtype=int), df_fix['yAvg'].to_numpy(dtype=int), df_fix['duration'].to_numpy()
    circles = draw_circles(ax, xs, ys, ts, df_fix, min_t, fix_size, ann_size)
    arrows = draw_arrows(ax, circles)
    hlines = draw_hlines(ax, lines_coords)
    buttons = draw_buttons(ax, img.shape) if editable else []

    cids = []
    if editable:
        last_actions = []
        cids.append(fig.canvas.mpl_connect('button_press_event',
                                           lambda event: onclick(event, circles, arrows, fig, ax, last_actions,
                                                                 df_fix, lines_coords, hlines, buttons)))
        cids.append(fig.canvas.mpl_connect('motion_notify_event',
                                           lambda event: move_object(event, ax, arrows, circles, last_actions)))
        cids.append(fig.canvas.mpl_connect('button_release_event',
                                           lambda event: release_object(event, lines_coords, df_fix, last_actions)))

    ax.axis('off')
    fig.canvas.draw()

    return cids


def draw_buttons(ax, img_shape, button_size=20):
    """Draw arrow buttons similar to how circles are drawn"""
    height, width = img_shape[:2]
    button_x = width - 30
    up_button_y = height * 0.4
    down_button_y = height * 0.6

    buttons = []
    up_circle = mpl.patches.Circle((button_x, up_button_y),
                                   radius=button_size,
                                   alpha=0.7,
                                   edgecolor='blue',
                                   linewidth=1.0)
    ax.add_patch(up_circle)
    up_annotation = plt.annotate("▲", xy=(button_x, up_button_y), fontsize=20, ha="center",
                                 va="center", color='blue', alpha=0.7)
    up_button = ArrowButton(0, up_circle, up_annotation, 'up')
    buttons.append(up_button)
    down_circle = mpl.patches.Circle((button_x, down_button_y),
                                     radius=button_size,
                                     alpha=0.7,
                                     edgecolor='red',
                                     linewidth=1.0)
    ax.add_patch(down_circle)
    down_annotation = plt.annotate("▼", xy=(button_x, down_button_y), fontsize=20, ha="center",
                                   va="center", color='red', alpha=0.7)
    down_button = ArrowButton(1, down_circle, down_annotation, 'down')
    buttons.append(down_button)

    return buttons


def draw_circles(ax, xs, ys, ts, df_fix, min_t, fix_size, ann_size):
    colors = mpl.colormaps['rainbow'](np.linspace(0, 1, xs.shape[0]))
    circles = []
    for i, (x, y, t) in enumerate(zip(xs, ys, ts)):
        aug_factor = 1 if t <= min_t else t / min_t
        radius = int(fix_size * aug_factor)
        circle = mpl.patches.Circle((x, y),
                                    radius=radius,
                                    color=colors[i],
                                    alpha=0.3)
        ax.add_patch(circle)
        fixation = df_fix.iloc[i]
        annotation = plt.annotate("{}".format(fixation.name + 1), xy=(x, y), fontsize=ann_size, ha="center",
                                  va="center", alpha=0.5)
        fix_circle = FixCircle(i, circle, annotation, fixation)
        circles.append(fix_circle)
    return circles


def draw_arrows(ax, circles):
    arrows = [draw_arrow(ax, circles[i].center(), circles[i + 1].center(), circles[i].color())
              for i in range(len(circles) - 1)]
    return arrows


def draw_hlines(ax, lines_coords):
    hlines = []
    if lines_coords is not None:
        for i, line_coord in enumerate(lines_coords):
            line2d = ax.axhline(y=line_coord, color='black', lw=0.5)
            line = HLine(i, line2d)
            hlines.append(line)
    return hlines


def draw_arrow(ax, p1, p2, color, alpha=0.2, width=0.05):
    x1, y1 = p1
    x2, y2 = p2
    arrow = mpl.patches.Arrow(x1, y1, x2 - x1, y2 - y1, width=width, color=color, alpha=alpha)
    ax.add_patch(arrow)
    return arrow


def move_horizontal_lines(hlines, lines_coords, offset):
    if lines_coords is None or len(lines_coords) <= 1:
        return

    for i in range(1, len(lines_coords)):
        lines_coords[i] += offset

    for i in range(1, len(hlines)):
        current_y = hlines[i].line.get_ydata()[0]
        hlines[i].line.set_ydata([current_y + offset, current_y + offset])


def screen(points=[], point_size=14, height=1080, width=1920, color='grey'):
    """ Draw an empty image with the given points """
    img = Image.new('RGB', (width, height), color=color)
    draw = ImageDraw.Draw(img)
    if len(points):
        xs, ys = points['x'].to_numpy(), points['y'].to_numpy()
        for x, y in zip(xs, ys):
            draw.ellipse((x - point_size, y - point_size, x + point_size, y + point_size), fill='black')

    return np.array(img)
