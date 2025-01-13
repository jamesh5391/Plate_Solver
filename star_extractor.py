import cv2
import itertools
import numpy as np
import math

# Load the image
image = cv2.imread('testphotos/stars.jpg')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#create filter for pixel intensity values < 210 
mask = cv2.inRange(gray_image, 240, 255)

#filter out less brighter pixels in image

result = cv2.bitwise_and(gray_image, gray_image, mask=mask)

res, threshold = cv2.threshold(result, 220, 255, 0); 

#get contours for stars 
contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


coords = []
for contour in contours:

    #find centroid coordinate of each eccentric star  
    M = cv2.moments(contour)

    x,y = contour[0,0]
    coords.append((x,y))

    cv2.circle(threshold, (x, y), 1, (255, 255, 255), -1)

# Display the result

cv2.imshow('Detected Stars', threshold)




#implement k-d tree for finding 3 closest stars to each star 

class KDTree(object):
    
    """
    The points can be any array-like type, e.g: 
        lists, tuples, numpy arrays.

    Usage:
    1. Make the KD-Tree:
        `kd_tree = KDTree(points, dim)`
    2. You can then use `get_knn` for k nearest neighbors or 
       `get_nearest` for the nearest neighbor

    points are be a list of points: [[0, 1, 2], [12.3, 4.5, 2.3], ...]
    """
    def __init__(self, points, dim, dist_sq_func=None):
        """Makes the KD-Tree for fast lookup.

        Parameters
        ----------
        points : list<point>
            A list of points.
        dim : int 
            The dimension of the points. 
        dist_sq_func : function(point, point), optional
            A function that returns the squared Euclidean distance
            between the two points. 
            If omitted, it uses the default implementation.
        """

        if dist_sq_func is None:
            dist_sq_func = lambda a, b: sum((x - b[i]) ** 2 
                for i, x in enumerate(a))
                
        def make(points, i=0):
            if len(points) > 1:
                points.sort(key=lambda x: x[i])
                i = (i + 1) % dim
                m = len(points) >> 1
                return [make(points[:m], i), make(points[m + 1:], i), 
                    points[m]]
            if len(points) == 1:
                return [None, None, points[0]]
        
        def add_point(node, point, i=0):
            if node is not None:
                dx = node[2][i] - point[i]
                for j, c in ((0, dx >= 0), (1, dx < 0)):
                    if c and node[j] is None:
                        node[j] = [None, None, point]
                    elif c:
                        add_point(node[j], point, (i + 1) % dim)

        import heapq
        def get_knn(node, point, k, return_dist_sq, heap, i=0, tiebreaker=1):
            if node is not None:
                dist_sq = dist_sq_func(point, node[2])
                dx = node[2][i] - point[i]
                if len(heap) < k:
                    heapq.heappush(heap, (-dist_sq, tiebreaker, node[2]))
                elif dist_sq < -heap[0][0]:
                    heapq.heappushpop(heap, (-dist_sq, tiebreaker, node[2]))
                i = (i + 1) % dim
                # Goes into the left branch, then the right branch if needed
                for b in (dx < 0, dx >= 0)[:1 + (dx * dx < -heap[0][0])]:
                    get_knn(node[b], point, k, return_dist_sq, 
                        heap, i, (tiebreaker << 1) | b)
            if tiebreaker == 1:
                return [(-h[0], h[2]) if return_dist_sq else h[2] 
                    for h in sorted(heap)][::-1]

        def walk(node):
            if node is not None:
                for j in 0, 1:
                    for x in walk(node[j]):
                        yield x
                yield node[2]

        self._add_point = add_point
        self._get_knn = get_knn 
        self._root = make(points)
        self._walk = walk

    def __iter__(self):
        return self._walk(self._root)
        
    def add_point(self, point):
        """Adds a point to the kd-tree.
        
        Parameters
        ----------
        point : array-like
            The point.
        """
        if self._root is None:
            self._root = [None, None, point]
        else:
            self._add_point(self._root, point)

    def get_knn(self, point, k, return_dist_sq=True):
        """Returns k nearest neighbors.

        Parameters
        ----------
        point : array-like
            The point.
        k: int 
            The number of nearest neighbors.
        return_dist_sq : boolean
            Whether to return the squared Euclidean distances.

        Returns
        -------
        list<array-like>
            The nearest neighbors. 
            If `return_dist_sq` is true, the return will be:
                [(dist_sq, point), ...]
            else:
                [point, ...]
        """
        return self._get_knn(self._root, point, k, return_dist_sq, [])

    def get_nearest(self, point, return_dist_sq=True):
        """Returns the nearest neighbor.

        Parameters
        ----------
        point : array-like
            The point.
        return_dist_sq : boolean
            Whether to return the squared Euclidean distance.

        Returns
        -------
        array-like
            The nearest neighbor. 
            If the tree is empty, returns `None`.
            If `return_dist_sq` is true, the return will be:
                (dist_sq, point)
            else:
                point
        """
        l = self._get_knn(self._root, point, 1, return_dist_sq, [])
        return l[0] if len(l) else None

kd_tree = KDTree(coords, 2)
hashes = []
for coord in coords: 

    coordinates = kd_tree.get_knn(coord, 4, False)

    #print out quads for testing
    pts = np.array(coordinates, dtype=np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.polylines(threshold, [pts], isClosed=True, color=(255,255,255), thickness=2)

    pairs = list(itertools.combinations(coordinates, 2))
    hash = []
    maxVal = -1
    #find distances between all points of single quad
    for pair in pairs: 
        dist = math.sqrt(math.pow(pair[0][0] - pair[0][1], 2) + math.pow(pair[1][0] - pair[1][1], 2))
        if dist > maxVal: 
            maxVal = dist 
        hash.append(dist)
    
    ## TODO: Maybe Limit Number of Quads to N <= 300 
    #compute hash for each quad 
    hash = [x / maxVal for x in hash]
    hashes.append(sorted(hash))

print(hashes)
#print quads for testing
cv2.imshow("Quads", threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()

    
    




"""
combinations = list(itertools.combinations(coords, 4))
quads = []
for combo in combinations: 
    pairs = list(itertools.combinations(combo, 2))
    distances = []
    for pair in pairs: 
        distances.append(math.sqrt(math.pow(pair[0][0] - pair[0][1], 2) + math.pow(pair[1][0] - pair[1][1], 2)))
"""