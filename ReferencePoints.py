import numpy as np
from scipy.optimize import minimize


def search_minimum(fun, x):
    res = minimize(fun, x, method='Nelder-Mead')
    return res.x


class ReferencePoints:
    """
    Class for manipulating reference points.
    """

    def __init__(self, xy_holes):
        th_quality_square = 0.05
        side_square_mm = 1
        self.xy_holes = xy_holes
        self.side_square_mm = side_square_mm
        self.quality_square = 0
        self.pars = {"th_quality_square": th_quality_square}
        self.transf = {}
        self.error = {"errorList": [], "errorMeaningList": []}
        self.aux = {}

        if len(self.xy_holes) == 4:
            self.quality_square = self.posRef2qualitySquare(self.xy_holes)
            x0, v1, v2 = self.posRef2base(self.xy_holes)
            x0 = x0 + v1 / 2 + v2 / 2
            mmPerPixel = self.side_square_mm / np.linalg.norm(v1)
            v1 = v1 / np.linalg.norm(v1)
            v2 = v2 / np.linalg.norm(v2)
            self.transf = {"x0": x0, "v1": v1, "v2": v2, "mmPerPixel": mmPerPixel}

            if self.quality_square > self.pars["th_quality_square"]:
                self.addError(1.2, "Reference points not usable: Quality not good enough", show=True)
        else:
            self.addError(1, "Reference points not usable: They are not 4.", show=True)

    def pixel2mm(self, xy):
        xy = xy - self.transf["x0"]
        xy = np.column_stack((np.sum(xy * self.transf["v1"], axis=1), np.sum(xy * self.transf["v2"], axis=1)))
        xy = xy * self.transf["mmPerPixel"]
        return xy

    def mm2pixel(self, xy):
        xy = xy / self.transf["mmPerPixel"]
        xy = np.column_stack((xy[:, 0] * self.transf["v1"] + xy[:, 1] * self.transf["v2"]))
        xy = xy + self.transf["x0"]
        return xy

    def addError(self, codeError, strError, show=False):
        self.error["errorList"].append(codeError)
        self.error["errorMeaningList"].append(strError)

        if show:
            print(strError)

    def posRef2base(self, posRef, show=False):
        orderY = np.argsort(posRef[:, 1])
        orderFinal = np.empty(4, dtype=int)

        for iRow in range(2):
            indPointRow = orderY[iRow * 2: (iRow + 1) * 2]
            orderX = np.argsort(posRef[indPointRow, 0])
            orderFinal[iRow * 2: (iRow + 1) * 2] = indPointRow[orderX]

        posRef = posRef[orderFinal]
        x0 = posRef[0]
        v1 = np.diff(posRef[0:2], axis=0)

        x0v1 = np.concatenate((x0, v1))
        x0v1 = search_minimum(lambda x: self.funError(x[0:2], x[2:4], posRef), x0v1)
        x0 = x0v1[0:2]
        v1 = x0v1[2:4]
        v2 = np.array([-v1[1], v1[0]])

        if show:
            plt.plot(posRef[:, 0], posRef[:, 1], ".")
            posTheor = np.array([x0, x0 + v1, x0 + v2, x0 + v1 + v2])
            plt.plot(posTheor[:, 0], posTheor[:, 1], "o")
            plt.axis("equal")
            plt.show()

        return x0, v1, v2

    def funError(self, x0, v1, posRef):
        v2 = np.array([-v1[1], v1[0]])
        err = np.sum((posRef[0] - x0) ** 2 + (posRef[1] - (x0 + v1)) ** 2 +
                     (posRef[2] - (x0 + v2)) ** 2 + (posRef[3] - (x0 + v1 + v2)) ** 2)
        return err

    def posRef2qualitySquare(self, posRef):
        qualitySquare = 0
        distMat = np.sum((posRef[:, None, :] - posRef) ** 2, axis=2)

        for iPoint in range(4):
            order = np.argsort(distMat[iPoint])
            vec1 = posRef[iPoint] - posRef[order[1]]
            vec1 = vec1 / np.linalg.norm(vec1)
            vec2 = posRef[iPoint] - posRef[order[2]]
            vec2 = vec2 / np.linalg.norm(vec2)
            qualitySquare += abs(vec1.dot(vec2))

        return qualitySquare
