import matplotlib.pyplot as plt
import matplotlib.lines
from matplotlib.transforms import Bbox, TransformedBbox
from matplotlib.legend_handler import HandlerBase
from matplotlib.image import BboxImage
import os
import find_data as fd
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
from Parameters import parameters as param

# Code from https://stackoverflow.com/questions/42155119/replace-matplotlib-legends-labels-with-image
# Allows to create legends with images instead of text beside the lines
class HandlerLineImage(HandlerBase):

    def __init__(self, image_name, space=15, offset=10):
        self.space = space
        self.offset = offset
        self.image_data = plt.imread(os.path.dirname(os.path.realpath(__file__)) + "/"*fd.is_linux() + "\\"*(not fd.is_linux()) + image_name)
        super(HandlerLineImage, self).__init__()

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        l = matplotlib.lines.Line2D([xdescent + self.offset, xdescent + (width - self.space) / 3. + self.offset],
                                    [ydescent + height / 2., ydescent + height / 2.])
        l.update_from(orig_handle)
        l.set_clip_on(False)
        l.set_transform(trans)

        bb = Bbox.from_bounds(xdescent + (width + self.space) / 3. + self.offset,
                              ydescent,
                              height * self.image_data.shape[1] / self.image_data.shape[0],
                              height)

        tbb = TransformedBbox(bb, trans)
        image = BboxImage(tbb)
        image.set_data(self.image_data)

        self.update_prop(image, orig_handle, legend)
        return [l, image]

def distance_x_labels(condition_list, ax):
    # Set the x labels to the distance icons!
    # Stolen from https://stackoverflow.com/questions/8733558/how-can-i-make-the-xtick-labels-of-a-plot-be-simple-drawings
    for i in range(len(condition_list)):
        ax.set_xticks([])
        # Image to use
        arr_img = plt.imread(fd.return_icon_path(param.nb_to_distance[condition_list[i]]))
        # Image box to draw it!
        imagebox = OffsetImage(arr_img, zoom=0.8)
        imagebox.image.axes = ax
        x_annotation_box = AnnotationBbox(imagebox, (i, 0),
                                          xybox=(0, -8),
                                          # that's the shift that the image will have compared to (i, 0)
                                          xycoords=("data", "axes fraction"),
                                          boxcoords="offset points",
                                          box_alignment=(.5, 1),
                                          bboxprops={"edgecolor": "none"})
        ax.add_artist(x_annotation_box)
    return ax

