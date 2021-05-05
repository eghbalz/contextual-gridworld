import numpy as np

COLORS = {
    'black': np.array([0, 0, 0]),  # don't use
    'red': np.array([255, 0, 0]),
    'darkgreen': np.array([0, 100, 0]),
    'rosybrown': np.array([188, 143, 143]),
    'orangered': np.array([255, 96, 0]),
    'gold': np.array([255, 215, 0]),
    'lime': np.array([0, 255, 0]),
    'royalblue': np.array([65, 105, 225]),
    'aqua': np.array([0, 255, 255]),
    'blue': np.array([0, 0, 255]),
    'deeppink': np.array([255, 20, 147]),
    'saddlebrown': np.asarray([139, 69, 19]),
    'forestgreen': np.asarray([34, 139, 34]),
    'steelblue': np.asarray([70, 130, 180]),
    'darkblue': np.asarray([0, 0, 139]),
    'fuchsia': np.asarray([255, 0, 255]),
    'moccasin': np.asarray([255, 228, 181]),
    'hotpink': np.asarray([255, 105, 180]),
    'grey': np.array([100, 100, 100]),  # wall color
    'green': np.array([0, 255, 0]),
    'purple': np.array([112, 39, 195]),
    'yellow': np.array([255, 255, 0]),
    'pink': np.array([255, 20, 147]),
    'brown': np.array([165, 42, 42]),
    'cyan': np.array([0, 255, 255]),
    'teal': np.array([0, 128, 128]),
    'plum': np.array([221, 160, 221]),
    'orange': np.array([255, 165, 0]),
    'navy': np.array([0, 0, 128]),
    'khaki': np.array([240, 230, 140]),
    'olive': np.array([128, 128, 0]),
    'maroon': np.array([128, 0, 0]),
    'pale_violet_red': np.array([219, 112, 147]),
    'dark_olive_green': np.array([85, 107, 47]),
    'lavender': np.array([230, 230, 250]),
    'rosy_brown': np.array([188, 143, 143])
}

COLOR_NAMES = sorted(list(COLORS.keys()))
COLOR_TO_IDX = {k: v for v, k in enumerate(COLORS.keys())}
IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))
