import matplotlib.pyplot as plt
import matplotlib.lines
from matplotlib.transforms import Bbox, TransformedBbox
from matplotlib.legend_handler import HandlerBase
from matplotlib.image import BboxImage
import os
import find_data as fd

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
