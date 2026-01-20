from dolfin import interpolate, Expression, Function, Constant
import numpy as np

class CookieField:
    """
    A parametrized field to encode parameters for the cookie problem
    Diffusion coefficient is one value inside the circles and another outside them
    the circles can have individual radii or all the same radius,
    which influences the parameter dimension
    """
    def __init__(self, num_disks_per_dim, mean=1.0, radius_range=(0.5, 0.9), fixed_radii=True):
        """
        :param num_disks_per_dim:
        :param mean:
        :param radius_range:
        :param fixed_radii:
        """
        self.fixed_radii = fixed_radii
        self.num_disks_per_dim = num_disks_per_dim
        if fixed_radii:
            self.M = num_disks_per_dim**2
            assert radius_range[0] == radius_range[1]
        else:
            self.M = 2 * num_disks_per_dim**2

        self.mean = mean
        self.radius_range = radius_range
        self.a = Expression(
            """
            (pow(x[0] - center_x, 2.0) + pow(x[1] - center_y, 2.0) <= pow(radius, 2.0)) ? 1.0 : 0.0;
            """,
            radius=0, center_x=0, center_y=0, degree=5
        )

    def realisation(self, y, V):  # type: (List[float], FunctionSpace) -> Function
        """
        Compute an interpolation of the cookie field on the FEM space V for the given sample y.

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
        assert len(y) == self.M

        out_vector = Function(V).vector().get_local() + self.mean
        indices = [(i, j) for i in range(self.num_disks_per_dim) for j in range(self.num_disks_per_dim)]
        for k, (i, j) in enumerate(indices):
            self.a.center_x = (i + 0.5) / self.num_disks_per_dim
            self.a.center_y = (j + 0.5) / self.num_disks_per_dim
            radius = 0 if self.fixed_radii else y[k + self.num_disks_per_dim**2]
            radius = radius * (self.radius_range[1] - self.radius_range[0]) + self.radius_range[0]
            radius /= self.num_disks_per_dim * 2
            self.a.radius = radius
            out_vector += interpolate(self.a, V).vector().get_local() * (y[k] + 1) * 0.5

        f = Function(V)
        f.vector().set_local(out_vector)
        return f

