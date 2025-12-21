# that's the lab to create the kd trees
import numpy as np
import numpy.linalg
import LPEG as lp
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import pandas as pd

def plot_boundary(dataset, new_point, knn_set, boundarys_list):
    fig, ax = plt.subplots()
    ax.scatter(dataset[:, 0], dataset[:, 1], c=dataset[:, 2])
    ax.scatter(new_point[0], new_point[1], color='red', label='query')
    ax.scatter(knn_set[:, 0], knn_set[:, 1], marker='+', color='red', label='nn')

    # compute plotting extents with small padding
    if dataset.shape[1] >= 2:
        x_min, x_max = dataset[:, 0].min(), dataset[:, 0].max()
        y_min, y_max = dataset[:, 1].min(), dataset[:, 1].max()
    else:
        x_min, x_max = -1, 1
        y_min, y_max = -1, 1
    x_pad = (x_max - x_min) * 0.05 if x_max > x_min else 0.1
    y_pad = (y_max - y_min) * 0.05 if y_max > y_min else 0.1
    x_min -= x_pad; x_max += x_pad; y_min -= y_pad; y_max += y_pad

    # plot each KD boundary (each item is {feature: value})
    colors = plt.cm.tab10.colors
    for i, b in enumerate(boundarys_list):
        for feat, val in b.items():
            col = colors[i % len(colors)]
            if feat == 0:
                # vertical line at x = val
                ax.plot([val, val], [y_min, y_max], color=col, linestyle='-', linewidth=1.5, alpha=0.9)
            else:
                # horizontal line at y = val
                ax.plot([x_min, x_max], [val, val], color=col, linestyle='-', linewidth=1.5, alpha=0.9)

    ax.set_aspect('equal', adjustable='datalim')
    ax.legend()
    plt.show()

class Node:
    def __init__(self, l_child = None, r_child = None, full_connected = False):
        """
        Parameters
        ----------
        center : np.ndarray[float]
            the center of the ball with the same dimension of the data
        """
        self.parent = None
        self.r_child = r_child
        self.l_child = l_child
        if full_connected:
            self.r_child.parent = self
            self.l_child.parent = self

class Ball_node(Node):
    def __init__(self, center : np.ndarray[float], radius : float, l_child = None, r_child = None, full_connected = False):
        """
        Parameters
        ----------
        center : np.ndarray[float]
            the center of the ball with the same dimension of the data
        """
        super().__init__( l_child, r_child, full_connected)
        self.__center = center
        self.__radius = radius
    
    def center(self):
        return self.__center
    
    def radius(self):
        return self.__radius

class Kd_node(Node):
    def __init__(self, feature, value : float, l_child = None, r_child = None, full_connected = False):
        super().__init__( l_child, r_child, full_connected)
        self.feature = feature
        self.value = value

class Data_leaf:
    def __init__(self, data):
        """
        Parameters
        ----------
        datas : array like
            the set of all the data related to that leaf
        """
        self.parent = None
        self.data = data
        print("\n",data)

class Bs_tree:
    def __init__(self, knn: int = 1, distance_metric : callable = None):
        """
        Parameters
        ----------
        knn : int
            number of max dimension of the leafs
        distance : function(point1, point2) -> float
            a function that compute the distance between two points
        """
        self.knn = knn
        self.distance_metric = distance_metric
        if distance_metric == None:
            self.distance_metric = np.linalg.norm
    
    def distance(self, set1: np.ndarray, set2 : np.ndarray):
        """
        This function compute the distance between two points or a set and a point
        
        Parameters
        ----------
        set1 : np.ndarray
            tested with 1 and 2 dimensional array
        set2 : np.ndarray
            tested only with 1 dimensional array
        
        """
        return self.distance_metric(np.full(set1.shape, set2) - set1, axis=1)

class Ball_tree(Bs_tree):
    def __init__(self, dataset, knn: int = 1, distance_metric : callable = None):
        """
        Parameters
        ----------
        knn : int
            number of max dimension of the leafs
        distance : function(point1, point2) -> float
            a function that compute the distance between two points
        """
        super().__init__(knn, distance_metric)
        self.root = self._tree_creator(dataset)

    def _tree_creator(self, dataset : np.ndarray):#TODO devo passare il dataset completo perché i punti devono tenere traccia della loro label
        if len(dataset) <= self.knn:
            return Data_leaf(dataset)
        
        i = np.random.randint(0, len(dataset))
        maxindex1 = np.argmax( self.distance(dataset, dataset[i]))
        point1 = dataset[maxindex1]
        maxindex2 = np.argmax( self.distance(dataset, point1))
        point2 = dataset[maxindex2]
        
        projected_points = dataset@(point2 - point1).T
        median = np.median(projected_points)
        l_dataset = dataset[projected_points<median]
        r_dataset = dataset[projected_points>=median]

        center = np.mean(dataset, axis=0)
        root = Ball_node( center = center,
                          radius = np.max(self.distance(dataset, center)),
                          l_child = self._tree_creator(l_dataset),
                          r_child = self._tree_creator(r_dataset),
                          full_connected = True
                          )        
        return root

def plot_ball_tree(tree, ax=None, show=True, circle_kwargs=None, point_kwargs=None): # copilot
    """ 
    Disegna il dataset (solo 2D) e tutte le circonferenze dei `Ball_node` contenute
    in un `Ball_tree` costruito con questa implementazione.

    Parametri
    ---------
    tree : Ball_tree
        L'albero delle palla da disegnare (deve avere `root` come attributo).
    ax : matplotlib.axes.Axes, optional
        Axes su cui disegnare. Se None viene creato un nuovo plot.
    show : bool
        Se True chiama `plt.show()` alla fine.
    circle_kwargs : dict, optional
        Argomenti passati a `matplotlib.patches.Circle` (es. edgecolor, linewidth, alpha).
    point_kwargs : dict, optional
        Argomenti passati a `ax.scatter` per disegnare i punti.

    Restituisce
    ---------
    ax : matplotlib.axes.Axes
        L'axes utilizzato per il disegno.
    """


    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    circle_kwargs = {} if circle_kwargs is None else dict(circle_kwargs)
    point_kwargs = {} if point_kwargs is None else dict(point_kwargs)

    centers = []
    radii = []
    points_list = []

    def traverse(node):
        if isinstance(node, Data_leaf):
            if getattr(node, 'data', None) is not None and len(node.data) > 0:
                pts = np.asarray(node.data)
                if pts.ndim == 1:
                    pts = pts.reshape(1, -1)
                points_list.append(pts)
            return
        if isinstance(node, Ball_node):
            centers.append(node.center())
            radii.append(node.radius())
            if getattr(node, 'l_child', None) is not None:
                traverse(node.l_child)
            if getattr(node, 'r_child', None) is not None:
                traverse(node.r_child)
            return
        # fallback: altri tipi di nodo (es. Kd_node)
        if getattr(node, 'l_child', None) is not None:
            traverse(node.l_child)
        if getattr(node, 'r_child', None) is not None:
            traverse(node.r_child)

    traverse(tree.root)

    if len(points_list) > 0:
        pts = np.vstack(points_list)
        if pts.shape[1] != 2:
            raise ValueError("plot_ball_tree: supporta solo dati 2D.")
        ax.scatter(pts[:, 0], pts[:, 1], s=20, **point_kwargs)

    for c, r in zip(centers, radii):
        c = np.asarray(c)
        if c.size != 2:
            continue
        circ = Circle((c[0], c[1]), r, fill=False, **circle_kwargs)
        ax.add_patch(circ)
        ax.plot(c[0], c[1], marker='+', color=circle_kwargs.get('edgecolor', 'C1'))

    ax.set_aspect('equal', adjustable='datalim')
    ax.autoscale_view()
    if show:
        plt.show()
    return ax

class Kd_tree(Bs_tree):
    def __init__(self, dataset, knn: int = 1, distance_metric : callable = None, decision_function : callable = None):
        """
        Parameters
        ----------
        knn : int
            number of max dimension of the leafs
        distance : function(point1, point2) -> float
            a function that compute the distance between two points
        """
        super().__init__(knn, distance_metric)
        if decision_function is None:
            self.decision_fun = lambda a,b : a < b
        self.root = self._tree_creator(dataset)

    def _correct_child(self, node : Kd_node, point : np.ndarray):
        if self.decision_fun(point[node.feature], node.value):
            return node.l_child
        return node.r_child

    def _tree_creator(self, dataset : np.ndarray):
        if len(dataset) <= self.knn:
            return Data_leaf(dataset)
        
        feature = np.random.randint(0, dataset.shape[1]-1 )
        median = np.median(dataset[:, feature])
        l_dataset = dataset[dataset[:, feature]<median]
        r_dataset = dataset[dataset[:, feature]>=median]

        root = Kd_node( feature = feature,
                        value= median,
                        l_child = self._tree_creator(l_dataset),
                        r_child = self._tree_creator(r_dataset),
                        full_connected = True
                        )
        return root
    
    def _tree_descend(self, initial_node : Kd_node | Data_leaf, point : list):
        """
        Desend the tree to find the nearest datas
        
        Parameters
        ----------
        knn : int
            number of max dimension of the leafs
        distance : function(point1, point2) -> float
            a function that compute the distance between two points
        
        Return
        ----------
        node.data : list[data]
            return the datas founded in the closest leaf of the tree
        boundary : list[dict]
            the list of the boundary founded will searching the leaf
        """
        boundary = []
        node = initial_node
        while not isinstance(node, Data_leaf):
            boundary.append( {node.feature : node.value})
            node = self._correct_child(node, point)
        return node.data, boundary
    
    def nearest_neighbor(self,point):
        # search nn
        node = self.root
        knn_set, boundary = self._tree_descend(node, point)

        # find se ci sono altri più vicini
        # calcolo la max distance
        max_distance = max(self.distance(knn_set, point))
        # risalgo l'albero e ogni volta valuto la distanza della boundary con la max distance
        # se trovo boundary più distanti ignoro e continuo a salire
        # se trovo boundary più vicina, scendo tutto l'altro ramo rispettando tutti gli altri vincoli a cercare dei nuovi candidati
        # quando trovo dei nuovi nodi, valuto la distanza di tutti, ordino per distanza e prendo quelli più vicini
        # torno a salire l'albero

        return knn_set, max_distance

dataset = lp.gaussian_clouds_data_generator(n = 200, sparcity=10, n_classes = 2, means=[0,0,2,2])
ball = Kd_tree(dataset, knn = 3) 
point = [1,1]
knn, boundary = ball.nearest_neighbor(point)
print("that's the inisial close\n", knn)
print(boundary)