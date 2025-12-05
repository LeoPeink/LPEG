import numpy as np

class DataSet:
    def __init__(self, data : np.ndarray, label, dim : int = None, polyGrade : int = 1):
        """_summary_

        Parameters
        ----------
        data (np.ndarray): _description_
        label (np.ndarray): _description_
        dim (int, optional): _description_. Defaults to None.
        polyGrade : int, optional
            used if have to operate with polinomial regression. Is the grade of the polinomy

        Raises:
            ValueError: 
        """
        self.x = data
        self.y = label
        self.dim = len(self.x[0]) if dim is None else dim if isinstance(dim, int) else None
        if self.dim is None:
            raise ValueError("Dimension have a unacceptable value")
        self.n = len(self.x)
        self.poly_grade = polyGrade
        
    def label(self, w : np.ndarray):
        """
        Return the labels of the dataset according to the weights w

        Parameters
        ----------
        w : np.ndarray
            w have to have lenght = dimension * (polyGrade + 1)
            

        Returns:
            _type_: _description_
        """
        polydim = self.poly_grade + 1
        ris = np.zeros(self.n)
        print(self.x)
        for i in range(self.dim):
            ris += np.polyval(w[i*polydim:(i+1)*polydim], self.x[:,i])
        return ris



x = np.array([[1,i] for i in range(10)])
y = [[w for w in range(10)]]
dataset = DataSet(x,y)
w = np.array([1,0,1,0])
print(dataset.label(w))