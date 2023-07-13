import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import copy as copy


class ReferencePoints:
    """
    Class for manipulating reference points.
    """

    def __init__(self, xy_holes):
        square_quality_threshold = 0.05  # If reference square has quality above this threshold, the plate will be rejected
        side_square_mm = 3.2  # Theoretical square size
        self.xy_holes = xy_holes  # coordinates of the four corners [[x0,y0],[x1,y1],[x2,y2],[x3,y3]]
        self.side_square_mm = side_square_mm  # Length of the side of the square of reference points, in mm
        self.square_quality = 0  # Quality of the square formed by the 4 reference points
        self.pars = {"square_quality_threshold": square_quality_threshold}
        self.transformation = {}  # Information needed to transform between the image (in pixels) and the aligned reference frame (in mm, rotated and with the origin at the center of the plate)
        self.errors = {"list_of_errors": [], "error_meaning_list": []}  # Information about errors

        # Fill the transformation dictionary
        if len(self.xy_holes) == 4:
            self.square_quality = compute_square_quality(self.xy_holes)
            x0, v1, v2 = references_to_base(self.xy_holes)  # x0 is the lower-left point, and v1, v2 have length equal to the square side
            # Translate base from lower left corner to center of plate
            x0 = x0 + v1 / 2 + v2 / 2
            # Scaling factor (determined by v1) (compute before normalizing vectors)
            mm_per_pixel = self.side_square_mm / np.linalg.norm(v1)
            # Renormalize the unit vectors so they have length 1 mm
            v1 = v1 / np.linalg.norm(v1)
            v2 = v2 / np.linalg.norm(v2)
            self.transformation = {"x0": x0, "v1": v1, "v2": v2, "mm_per_pixel": mm_per_pixel}

            if self.square_quality > self.pars["square_quality_threshold"]:
                self.add_error(1.2, "Reference points not usable: Quality not good enough", show=True)
        else:
            self.add_error(1, "Reference points not usable: They are not 4.", show=True)

    def pixel_to_mm(self, xy):
        """
        Transforms a set of points [[x y], [x y], [x y]], from the image reference system (in pixels) into the aligned ref. system (in mm).
        """
        # Subtract the center of plate coordinates
        xy = np.array(xy) - self.transformation["x0"]  # double brackets are because for 1D array np.transpose will return same array
        vector1 = self.transformation["v1"]
        vector2 = self.transformation["v2"]
        change_of_basis_matrix = np.linalg.inv(np.array([[vector1[0], vector2[0]], [vector1[1], vector2[1]]]))
        xy = np.transpose(np.dot(change_of_basis_matrix, np.transpose(xy)))
        xy = xy * self.transformation["mm_per_pixel"]
        return xy

    def mm_to_pixel(self, xy):
        """
         Transforms a set of points [[x x x], [y y y]], from the aligned ref. system (in mm) into the image reference system (in pixels).
        """
        xy = xy / self.transformation["mm_per_pixel"]
        vector1 = self.transformation["v1"]
        vector2 = self.transformation["v2"]
        change_of_basis_matrix = np.array([[vector1[0], vector2[0]], [vector1[1], vector2[1]]])
        xy = np.transpose(np.dot(change_of_basis_matrix, np.transpose(xy)))
        xy = xy + self.transformation["x0"]
        return xy

    def add_error(self, codeError, strError, show=False):
        """
        Adds an error to error property of the object.
        """
        self.errors["list_of_errors"].append(codeError)
        self.errors["error_meaning_list"].append(strError)
        if show:
            print(strError)


def search_minimum(fun, initial_guess):
    res = minimize(fun, initial_guess, method='Nelder-Mead')
    return res.x


def error_function(x0, v1, posRef):
    """
    Error function that is minimized in reference_to_base.
    """
    v2 = np.array([-v1[1], v1[0]])
    err = np.sum((posRef[0] - x0) ** 2 + (posRef[1] - (x0 + v1)) ** 2 +
                 (posRef[2] - (x0 + v2)) ** 2 + (posRef[3] - (x0 + v1 + v2)) ** 2)
    return err


def references_to_base(xy_holes, show=False):
    """
    I think this function gives you the two perpendicular vectors (v1 and v2)
    that best approximate the 4 reference points.
    """
    # Reorder points according to y then x, to get lower left corner then lower right then upper left then upper right
    xy_holes = sorted(xy_holes, key=lambda x: x[1])
    xy_holes = sorted(xy_holes[0:2], key=lambda x: x[0]) + sorted(xy_holes[2:4], key=lambda x: x[0])

    # Initial estimate for base, using only the two bottom points
    x0 = xy_holes[0]
    v1 = np.diff(xy_holes[0:2], axis=0)[0]

    # Minimization of error
    x0v1 = np.concatenate((x0, v1))  # Concatenate to make initial estimate
    x0v1 = search_minimum(lambda x: error_function(x[0:2], x[2:4], xy_holes), x0v1)
    x0 = x0v1[0:2]
    v1 = x0v1[2:4]
    v2 = np.array([-v1[1], v1[0]])

    if show:
        plt.plot([xy_holes[i][0] for i in range(len(xy_holes))], [xy_holes[i][1] for i in range(len(xy_holes))],
                 marker=".", label="Original holes")
        computed_positions = np.array([x0, x0 + v1, x0 + v2, x0 + v1 + v2])
        plt.plot(computed_positions[:, 0], computed_positions[:, 1], marker="o", label="Better square")
        plt.axis("equal")
        plt.legend()
        plt.show()

    return x0, v1, v2


def compute_square_quality(reference_points):
    """
    I think this function takes the coordinates of the 4 reference points (in xy_holes) and gives you a number that is 1
    if they form a perfect square, and lower if it's a shitty square.
    """
    square_quality = 0

    for i_point in range(4):
        current_point = reference_points[i_point]
        # List with points that are not the current point we're examining
        other_points = copy.deepcopy(reference_points)
        other_points.remove(current_point)
        # Go through the other points computing distance with current point
        distances = []
        for j_point in range(3):
            other_point = other_points[j_point]
            distances.append(np.sqrt((other_point[1] - current_point[1]) ** 2 + (other_point[0] - current_point[0]) ** 2))
        other_points.remove(other_points[np.argmax(
            distances)])  # remove point with max distance, because it is diagonally facing our point
        # Compute the dot product of the two vectors formed by our current point and the two others
        # This gives us a metric of how perpendicular they are (=> how square)
        vector_1 = np.array([other_points[0][0] - current_point[0], other_points[0][1] - current_point[1]])
        vector_1 = vector_1 / np.linalg.norm(vector_1)
        vector_2 = np.array([other_points[1][0] - current_point[0], other_points[1][1] - current_point[1]])
        vector_2 = vector_2 / np.linalg.norm(vector_2)
        square_quality += abs(np.dot(vector_1, vector_2))

    return square_quality


def test():
    ref_points = [[0, 0], [7, 1], [0, 7], [7, 8]]
    references_to_base(ref_points, show=True)
    ref_object = ReferencePoints(ref_points)
    new_ref_object = ReferencePoints([[0, 0], [1, 0], [0, 1], [1, 1]])
    points = [[3, 4], [3, 6]]
    points_in_mm = ref_object.mm_to_pixel(points)
    new_ref_points = new_ref_object.pixel_to_mm(points_in_mm)
    plt.scatter([3, 3], [4, 6], color="green", label="old points")
    plt.scatter([points_in_mm[0][0], points_in_mm[1][0]], [points_in_mm[0][1], points_in_mm[1][1]],
                color="black", label="points_mm")
    plt.scatter([new_ref_points[0][0], new_ref_points[1][0]], [new_ref_points[0][1], new_ref_points[1][1]],
                color="grey", label="new_points")
    plt.show()


#test()
