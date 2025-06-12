import math
from ..services.pedigree_tree import PedigreeTree

class BaseProcessor():
    def __init__(self,tree: PedigreeTree):
            self.tree = tree

    def get_bounding_box(self,detection: dict) -> list[int]:
        """
        get bounding box of top-left and bottom- right coodinates from detection_json
        """
        x = detection["x"]
        y = detection["y"]
        width = detection["width"]
        height = detection["height"]
        # Calculate top-left corner coordinates
        x1 = int(x - width / 2)
        y1 = int(y - height / 2)

        # Calculate bottom-right corner coordinates
        x2 = int(x + width / 2)
        y2 = int(y + height / 2)

        return [x1, y1, x2, y2]


    def overlap(self,boxA: list[int], boxB: list[int]) -> int:
        '''
        compute the overlapping area between two bounding boxes
        '''
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        inter_area = abs(max((xB - xA, 0)) * max((yB - yA), 0))
        return inter_area

    def relative_overlap(self,boxA: list[int], boxB: list[int]) -> int:
        '''
        calculate relative area of boxB overlapping with the boxA
        '''
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        inter_area = abs(max((xB - xA, 0)) * max((yB - yA), 0))
        boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        return inter_area/boxB_area




    def calculate_center(self,bbox):
        '''
        Calculate the center of a bounding box.
        '''
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        return (center_x, center_y)


    def euclidean_distance(self,point1, point2):
        '''
        Calculate the Euclidean distance between two points.
        '''
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

if __name__ == "__main__":
    pass
