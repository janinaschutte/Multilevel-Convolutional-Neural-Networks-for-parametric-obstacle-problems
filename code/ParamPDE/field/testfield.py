from dolfin import interpolate, Expression, Function, Constant
import numpy as np

class TestField:
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
    def __init__(self, M, k=[], mean=1.0, scale=1.0, decay=0.0, expfield=False):
        assert M > 0, 'number of terms in expansion has to be positive'
        assert isinstance(k, list)
        assert decay <= 0
        if not expfield:
            assert mean >= scale*sum(m**decay for m in range(1,M+1)), "{} vs {}".format(mean, scale*sum(m**decay for m in range(1,M+1)))
        # create a Fenics expression of the affine field
        self.a = Expression('sin(pi*F1*x[0]) * sin(pi*F2*x[1])', F1=0, F2=0, degree=5)
        self.M = M
        self.k = k
        self.mean = mean
        self.scale = scale
        self.decay = decay
        self.expfield = expfield

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
        out : ndarray
            FEM coefficients for the field realisation.
        """
        assert len(y) == self.M, 'number of parameters differs from the number of terms in the expansion'
        a = self.a
        k = self.k
        decay = self.decay
        scale = self.scale
        mean = self.mean

        def indexer(i):
            m1 = np.floor(i/2)
            m2 = np.ceil(i/2)
            return m1, m2

        #@cache  #TODO: klepto
        def summand(f1, f2):
            a.F1, a.F2 = f1, f2
            return interpolate(a, V).vector().get_local()

        def linear_part(ys, ms):
            x = Function(V).vector().get_local()  # zero
            assert np.all(x == 0)
            for m in ms:
                x += (m+1)**decay * ys[m] * summand(*indexer(m+2))
            x *= scale
            return x

        if not self.expfield:
            if len(k) == 0:
                x = mean + linear_part(y, range(self.M))
            elif len(k) == 1:
                x = linear_part(np.ones(self.M), k)
            else:
                x = linear_part(np.ones(self.M), [])
        else:
            x = np.exp(mean + linear_part(y, range(self.M)))
            for ki in k:
                x *= linear_part(np.ones(self.M), [ki])

        f = Function(V)
        f.vector().set_local(x)
        return f

        f.vector().set_local(a_val)
        return f
