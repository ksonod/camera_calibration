import matplotlib.pyplot as plt
import numpy as np


def show_cb_image_with_detected_corners(
        img: np.ndarray, detected_points: np.ndarray, figure_title: str, marker_style=".", marker_color="green"
):
    """
    This function shows a checkerboard image with detected points.

    :param img: image (gray or color).
    :param detected_points: Numpy array with the dimension of [number of detected points] x 2.
    :param figure_title: figure tile to be visualized.
    :param: marker_style: marker for showing detected checker corners.
    :param: marker_color: marker color
    """

    plt.imshow(img)
    if figure_title is not None:
        plt.title(figure_title)

    for idx_points in range(detected_points.shape[0]):
        plt.plot(
            detected_points[idx_points, 0], detected_points[idx_points, 1],
            marker=marker_style, color=marker_color
        )


def draw_XY_arrows(
        origin_point: np.ndarray,
        x0: np.ndarray,
        y0: np.ndarray,
        magnification_factor,
        head_width: int,
        head_length: int,
        arrow_color="red",
        annotation_color="yellow"
):
    """
    Draw X and Y axes arrows
    :param origin_point: Array with 2 elements for (X, Y) = (0, 0)
    :param x0: Array with 2 elements. X axis can be defined by x0 - origin_point.
    :param y0: Array with 2 elements. Y axis can be defined by x0 - origin_point.
    :param magnification_factor: Arrow magnification factor used for determining the length of arrows
    :param head_width: Arrow head width
    :param head_length: Arrow head length
    :param arrow_color: Arrow color
    :param annotation_color: Annotation ("X" and "Y") colors
    """

    # Set arrow lengths for visualization
    arrow_lengthX_x = magnification_factor * (x0[0] - origin_point[0])
    arrow_lengthX_y = magnification_factor * (x0[1] - origin_point[1])
    arrow_lengthY_x = magnification_factor * (y0[0] - origin_point[0])
    arrow_lengthY_y = magnification_factor * (y0[1] - origin_point[1])

    # Show an origin point
    plt.plot(origin_point[0], origin_point[1],
             color="red", marker="o", markersize=5, markerfacecolor='none')

    # Show an arrow in X direction.
    plt.arrow(x=origin_point[0], y=origin_point[1],
              dx=arrow_lengthX_x, dy=arrow_lengthX_y,
              head_width=head_width, head_length=head_length, color=arrow_color)
    plt.annotate(
        text="X",
        xy=(origin_point[0] + arrow_lengthX_x + 10, origin_point[1] + arrow_lengthX_y),
        color=annotation_color
    )

    # Show an arrow in Y direction.
    plt.arrow(x=origin_point[0], y=origin_point[1],
              dx=arrow_lengthY_x, dy=arrow_lengthY_y,
              head_width=head_width, head_length=head_length, color=arrow_color)
    plt.annotate(
        text="Y",
        xy=(origin_point[0] + arrow_lengthY_x, origin_point[1] + arrow_lengthY_y + 50),
        color=annotation_color
    )
