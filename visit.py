class Visit:
    def __init__(self, folder, start_index, start_frame, end_frame, patch):
        self.folder = folder
        self.index = start_index
        self.start = start_frame
        self.end = end_frame
        self.duration = start_frame - end_frame + 1
        self.patch = patch


class VisitList(list):
    def __init__(self):
        pass

