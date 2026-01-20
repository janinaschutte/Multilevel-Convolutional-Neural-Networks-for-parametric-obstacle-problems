from dolfin import interpolate, Expression, Function, Constant
from dolfin import *
import numpy as np

class RoughField:
    """
    Artificial M-term KL (plane wave Fourier modes similar to those in EGSZ).
    Diffusion coefficient reads (in linear case)

        a(x,y) = mean + scale * \sum_{m=1}^M (m+1)^decay y_m \sin(\pi b_1 x_1) \sin(\pi b_2 x_2)

    for b_1 = floor( (m+2)/2 ) and b_2 = ceil( (m+2)/2 ). In exponential case,

        b(x,y) = exp(a(x,y)).

    The field is log-normal for expfield=True.

    Parameters:
    -----------
    M : int
        Number of terms in expansion.
    k : int
        Modes to differentiate.
    mean : float, optional
        Mean value of the field.
        (defaults to 1.0)
    scale : float, optional
        Scaling coefficient for the centered terms of the expansion.
        (defaults to 1.0)
    decay : float, optional
        Decay rate for the terms.
        (defaults to 0.0)
    expfield : bool, optional
        Switch to choose lognormal field.
        (defaults to False)
    """
    def __init__(self):
        # self.a = Expression('cos(F1*x[0] + F2*x[1] + F3)', F1=0, F2=0, F3=0, degree=5) # on [-1,1]^2
        self.a = Expression('cos(F1*(x[0]*2-1) + F2*(x[1]*2-1) + F3)', F1=0, F2=0, F3=0, degree=5) # on [0,1]^2
        self.m = np.pi
        self.q_0 = 1
        self.q_s = 26
        self.q_l = 10
        self.q_multipliers = []
        bound = int(self.q_s//self.m)
        max_mult = range(-bound,bound+1)
        for i in max_mult:
            for j in max_mult:
                q = [self.m*i,self.m*j]
                if self.q_0<= np.linalg.norm(q) <= self.q_s:
                    self.q_multipliers.append(q)
        q_norms = np.array([np.linalg.norm(q) for q in self.q_multipliers])
        q_args = np.argsort(q_norms)
        self.q_multipliers = np.array(self.q_multipliers)[np.array([q for q in q_args]).astype(np.int)]
        print("Maximal number of random coefficients:", len(self.q_multipliers))
        print("Maximal norm:", np.max([np.linalg.norm(np.array(q)) for q in self.q_multipliers]))
        import matplotlib.pyplot as plt
        m = UnitSquareMesh(300,300)
        V = FunctionSpace(m, "CG", 1)
        y = 2*np.random.rand(len(self.q_multipliers))-1
        f = self.realisation(y, V)        
        c = plot(f)
        plt.colorbar(c)
        plt.savefig("code/FinalConvModel/Images/obstacle-rough/obstacle_example.png")
        plt.clf()

    def realisation(self, y, V):  # type: (List[float], FunctionSpace) -> Function
        """
        Compute an interpolation of the random field on the FEM space V for the given sample y.

        Parameters:
        -----------
        y : list of floats
            Parameters.
        V : FunctionSpace

        Returns
        -------
        out : FEM function
        """
        assert len(y)-1<len(self.q_multipliers)

        H = (y[0]+1)/2 # distributed as U([0,1)) # 1 #

        def BH(q):
            return np.pi*((2*np.pi*np.maximum(np.linalg.norm(q), self.q_l))**(-(H+1)))/16#25

        a = self.a
        phi_qs =  (y[1:]+1)*np.pi # distributed as U([0,2*pi)) # (y+1)*np.pi #
        #@cache  #TODO: klepto
        def shifted_cos(q, phi):
            a.F1, a.F2, a.F3 = q[0], q[1], phi
            return interpolate(a, V).vector().get_local()
        
        x = Function(V).vector().get_local()  # zero
        x = x - 0.5 # instead of 0.5 Dirichlet boundary condition
        for i,q in enumerate(self.q_multipliers[:len(phi_qs)]):
            phi_q = phi_qs[i]
            x += BH(q)*shifted_cos(q,phi_q)

        f = Function(V)
        f.vector().set_local(x)
        return f
