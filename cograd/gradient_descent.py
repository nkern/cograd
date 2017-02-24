"""
gradient_descent.py
-------------------

Algorithms for nonlinear conjugate gradient descent and
finite difference approximations to gradients

https://en.wikipedia.org/wiki/Nonlinear_conjugate_gradient_method
https://en.wikipedia.org/wiki/Finite_difference

--------------
Nicholas Kern
February, 2017
"""
import numpy as np
import scipy.linalg as la
import sklearn.gaussian_process as gp
import traceback

__all__ = ['FiniteDiff','Conj_Grad']

class FiniteDiff(object):

    def __init__(self):
        pass

    def first_central(self, f_neg, f_pos, dx):
        """
        Central finite difference for first order partial derivative of f

        Input:
        ------

        f_neg : scalar
                f(x-dx,y,z,..)

        f_pos : scalar
                f(x+dx,y,z,..)

        dx    : scalar
                dx

        Output:
        -------
        first_central : scalar
        """
        return (f_pos - f_neg)/(2*dx)

    def first_forward(self, f, f_pos, dx):
        """
        Forward finite difference for first order partial derivative of f

        Input:
        ------
        f     : scalar
                f(x,y,z,..)

        f_pos : scalar
                f(x+dx,y,z,..)

        dx    : scalar
                dx

        Output:
        -------
        first_forward : scalar
        """
        return (f_pos-f)/dx

    def second_central_cross(self, f, f_neg1, f_pos1, f_neg2, f_pos2, f_neg1_neg2, f_pos1_pos2, dx1, dx2):
        """
        Central finite difference approximation for second order partial derivative of f

        Input:
        ------

        f           : scalar
                    f(x,y,z,..)

        f_neg1      : scalar
                    f(x-dx1,y,z,..)

        f_pos1      : scalar
                    f(x+dx1,y,z,..)

        f_neg2      : scalar
                    f(x,y-dx2,z,..)

        f_pos2      : scalar
                    f(x,y+dx2,z,..)

        f_neg1_neg2 : scalar
                    f(x-dx1,y-dx2,z,..)

        f_pos1_pos2 : scalar
                    f(x+dx1,y+dx2,z,..)

        dx1         : scalar
                    dx1

        dx2         : scalar
                    dx2

        Output:
        -------
        second_central_cross : scalar
        """
        return (f_pos1_pos2 - f_pos1 - f_pos2 + 2*f - f_neg1 - f_neg2 + f_neg1_neg2) / (2*dx1*dx2)

    def second_central_auto(self, f, f_pos, f_neg, dx):
        """
        Calculate second derivative f_xx

        Input:
        ------

        f           : scalar

        f_pos       : scalar

        f_neg       : scalar

        dx          : scalar

        Output:
        -------
        second_central_auto : scalar
        """
        return (f_pos - 2*f + f_neg)/(dx**2)

    def calc_jacobian(self, f, theta, diff_vec, attach=True, grad_type='forward'):
        """
        Calculate the approximate Jacobian Matrix
        theta = [x, y, z, ...]
        diff_vec = [dx, dy, dz, ...]

        Input:
        ------

        f           : function object
                    The underlying function

        theta       : ndarray
                    Position in parameter space

        diff_vec    : ndarray [dtype=float, shape=(ndim,)]
                    vector containing dx offsets  for gradient calculation       

        attach      : bool [default=True]
                    if True attach result to class
                    else return result

        grad_type   : string [default='forward']
                    Type of gradient: ['forward', 'central']

        Output:
        -------
        J : Jacobian matrix ndarray
        """
        ndim = len(diff_vec)

        # Calc Partials
        f0 = f(theta)
        pos_vec, neg_vec = self.calc_partials(f, theta, diff_vec, second_order=False)

        # Construct Jacobian Matrix
        J = np.empty((1,ndim))
        for i in range(ndim):
            if grad_type == 'forward':
                J[0,i] = self.first_forward(f0, pos_vec[i], diff_vec[i])
            elif grad_type == 'central':
                J[0,i] = self.first_central(neg_vec[i], pos_vec[i], diff_vec[i])

        if attach == True:
            self.J = J
        else:
            return J

    def calc_hessian(self, f, pos_mat, neg_mat, diff_vec, out_jacobian=True, attach=True):
        """
        Calculate the approximate Hessian Matrix

        Input:
        ------

        f           : scalar
            evaluation of "f" at fiducial point

        theta       : ndarray [dtype=float, shape=(ndim,)]
            Fiducial point in parameter space

        pos_mat     : ndarray [dtype=float, shape=(ndim,ndim)]
            A matrix holding evaluations of "f" at x1+dx1, x2+dx2, ...
            Diagonal is f_ii = f(..,xi+dxi,..). Example: f_11 = f(x1+dx1, x2, x3, ..)
            Off-diagonal is f_ij = f(..,xi+dxi,..,xj+dxj,..). Example: f_12 = f(x1+dx1, x2+dx2, x3, ..)

        neg_mat     : ndarray [dtype=float, shape=(ndim,ndim)]
            A matrix holding evaluations of "f" at x1-dx1, x2-dx2, ...
            Same format as pos_mat

        diff_vec    : ndarray [dtype=float, shape=(ndim,)]
            A vector holding the step size for each dimension. Example: (dx1, dx2, dx3, ...) 

        out_jacobian    : bool [default=True]
            If True: output jacobian matrix as well as hessian matrix

        attach : bool [default=True]
            if True attach result to class
            else return result

        Output:
        -------
        H : Approximate Hessian Matrix, ndarray
        J : Approximate Jacobian matrix if out_jacobian is True
        """
        ndim = len(diff_vec)

        # Calculate Hessian Matrix via Finite Difference
        H = np.empty((ndim,ndim))
        if out_jacobian == True: J = np.empty((1,ndim))
        for i in range(ndim):
            for j in range(i, ndim):
                if i == j:
                    hess = self.second_central_auto(f, pos_mat[i,i], neg_mat[i,i], diff_vec[i])
                    H[i,i] = 1 * hess
                else:
                    hess = self.second_central_cross(f, neg_mat[i,i], pos_mat[i,i], neg_mat[j,j], 
                                                pos_mat[j,j], neg_mat[i,j], pos_mat[i,j], diff_vec[i], diff_vec[j])
                    H[i,j] = 1 * hess
                    H[j,i] = 1 * hess
                if out_jacobian == True and j==i: J[0,i] = self.first_central(neg_mat[i,i], pos_mat[i,i], diff_vec[i])

        if out_jacobian == True:
            if attach == True:
                self.H, self.J = H, J
            else:
                return H, J
        else:
            if attach == True:
                self.H = H
            else:
                return H

    def calc_partials(self, f, theta, diff_vec, second_order=True):
        """
        Use finite difference to calculate pos_mat and neg_mat,
        which are matrices of the function, f, evaluated at f(x+dx)
        or f(y+dy) or f(x+dx, y+dy) or f(x+dx, y) etc.
        theta = [x, y, ...]
        diff_vec = [dx, dy, ...]

        Input:
        ------

        f           : function object

        theta       : ndarray
                    Position in spacev

        diff_vec    : ndarray
                    Difference offsets for gradient calculation

        second_order    : bool [default=True]
                    Calculate off diagonal terms of pos/neg_mat for Hessian matrix

        Output:
        -------
        pos_mat, neg_mat : ndarray, ndarray
                        matrices containing f evaluated at theta +/- diff_vec
        """
        ndim = len(diff_vec)

        # Calculate positive and negative matrices
        pos_mat = np.empty((ndim,ndim))
        neg_mat = np.empty((ndim,ndim))
        for i in range(ndim):
            if second_order == True: second = ndim
            else: second = i+1
            for j in range(i,second):
                theta_pos   = theta + np.eye(ndim)[i] * diff_vec
                theta_neg   = theta - np.eye(ndim)[i] * diff_vec
                if j != i:
                    theta_pos   += np.eye(ndim)[j] * diff_vec
                    theta_neg   -= np.eye(ndim)[j] * diff_vec
                f_pos       = f(theta_pos)
                f_neg       = f(theta_neg)
                pos_mat[i,j] = 1 * f_pos
                neg_mat[i,j] = 1 * f_neg
                if i != j:
                    pos_mat[j,i] = 1 * f_pos
                    neg_mat[j,i] = 1 * f_neg

        if second_order == True:
            return pos_mat, neg_mat
        else:
            return pos_mat.diagonal(), neg_mat.diagonal()

    def propose_O2(self, H,J,gamma=0.5):
        """
        Give a second order proposal step from current position theta given Hessian H and Jacobian J
        In order to find local minima
        """
        # Evaluate proposal step
        prop = -np.dot(la.inv(H),J.T).ravel()

        return gamma * prop

    def propose_O1(self, J, gamma=0.5):
        """
        Give a first order proposal step to minimize a function
        """
        prop = -gamma*J
        return prop

    def find_root(self, f, theta, diff_vec, nsteps=10, gamma=0.1, second_order=False, bounds=None, step_size=None):
        """
        Find root

        f : function 
            a function that returns a scalar when passed an ndarray of shape theta

        theta : ndarray 
            starting point

        diff_vec : ndarray
            difference vector containing step size for gradient evaluation with shape theta

        nsteps : int (default=10)
            number of steps to take

        gamma : int or ndarray (default=0.1)
            learning rate: gradient coefficient

        second_order : bool (default=False)
            if True: use second order proposal
            if False: use first order proposal

        bounds : ndarray (default=None)
            hard bounds for search

        step_size : int or ndarray (default=None)
            if not None: force step size to of magnitude to be a fraction of bound extent: prop =  bound_size * step_size / np.abs(prop)
            if step_size.ndim == 2: prop = bound_sizes * step_size[i] / np.abs(prop)
        """
        # Get ndim
        ndim = len(diff_vec)

        # Check bounds and step_size
        if step_size is not None and bounds is None:
            print('If step_size is not None bounds must be not None')
            raise Exception

        # Get param size
        if bounds is not None:
            bound_sizes = np.abs(np.array(map(lambda x: x[1]-x[0], bounds)))

        # Iterate over nsteps
        steps = []
        grads = []
        for i in range(nsteps):
            try:
                # Append steps
                steps.append(np.copy(theta))

                # Approximate partial derivatives
                pos_mat, neg_mat = self.calc_partials(f, theta, diff_vec, second_order=second_order)
                if second_order == True:
                    H, J = self.calc_hessian(f(theta), pos_mat, neg_mat, diff_vec)
                    grads.append([self.H, self.J])
                else:
                    J = self.calc_jacobian(f(theta), pos_mat, diff_vec, neg_vec=neg_mat)
                    grads.append([self.J])

                # Compute proposal step
                if second_order == True:
                    prop = self.propose_O2(H, J[0], gamma=gamma)
                else:
                    prop = self.propose_O1(J[0], gamma=gamma)

                # Check if step_size is set
                if bounds is not None and step_size is not None:
                    if type(step_size) is np.ndarray:
                        if step_size.ndim == 1:
                            prop *= step_size * bound_sizes / np.abs(prop)
                        elif step_size.ndim == 2:
                            prop *= step_size[i] * bound_sizes / np.abs(prop)
                    elif type(step_size) is float or type(step_size) is int:
                        prop *= step_size * bound_sizes / np.abs(prop)

                # Check within bounds
                within = 1
                new_theta = 1*theta + prop
                for i in range(ndim):
                    if bounds is None:
                        pass
                    elif new_theta[i] < bounds[i][0] or new_theta[i] > bounds[i][1]:
                        within *= 0

                # Update position
                if within == 1:
                    theta = 1*new_theta
                else:
                    print('step #'+str(int(i))+' out of bounds')
                    theta = 1*theta - 0.1*prop

            except:
                print("Optimization failed... releasing steps already done")
                traceback.print_exc()
                return np.array(steps), np.array(grads)

        return np.array(steps), np.array(grads)

    def hess(self, theta):
        pos_mat, neg_mat = self.calc_partials(self.f, theta, self.diff_vec, second_order=True)
        H, J = self.calc_hessian(self.f(theta), pos_mat, neg_mat, self.diff_vec)
        return H

    def jac(self, theta):
        pos_vec, neg_vec = self.calc_partials(self.f, theta, self.diff_vec, second_order=False)
        J = self.calc_jacobian(self.f, pos_vec, self.diff_vec, neg_vec=neg_vec)
        return J[0]



class Conj_Grad(FiniteDiff):

    def __init__(self):
        """
        Exposed routines for the Nonlinear Conjugate Gradient algorithm,
        designed for when the underlying minimizable function (f) is expensive
        to evaluate and/or non-analytic (i.e. a complex computer simulation). 
        """
        pass

    def initialize(self, base_dir):
        """
        Initialize the communications between this script and the outside world

        Input:
        ------


        """
        pass

    def norm_gradient(self, f, x0, dx=0.01, grad_type='forward'):
        """
        Unit vector of gradient

        df/dx = (f(x) + f(x+dx))/dx
        return (df/dx) / norm(df/dx)

        Input:
        -------
        f       : function object
                Underlying target function to minimize

        x0      : ndarray 
                Starting point vector

        dx : ndarray or scalar [default=0.01]
                Offsets from center to compute numerical gradients

       grad_type   : string [default='forward']
                    Type of gradient: ['forward', 'central']

        Output:
        -------
        grad    : ndarray
                gradient unit vector
        """
        length = len(x0)
        if type(dx) != np.ndarray:
            dx = np.ones(length)*dx

        grad = -self.calc_jacobian(f, x0, dx, grad_type=grad_type, attach=False)[0]
        grad /= la.norm(grad)
        return grad
        
    def GP_line_search(self, f, x0, d, Nsample=50, distance_frac=0.5, backpace=0.0):
        """
        Gaussian Process Line Search

        Input:
        -------



        Output:
        -------


        Notes:
        ------
        The idea behind this algorithm is to sample a few points along the search direction and
        fit their response to a 1D GP, and then solve for its minimum via direct draws from the
        GP mean function along the search direction. To ensure we have not undersampled f we lay
        down another set of points in between the original set, fit another GP and compare the 
        the scale length hyperparameter, ell, to the original GP ell. We repeat until ell_2 is roughly
        the same as ell_1. Another stop point could be to stop when the derived minima are
        roughly the same. Both require an interpretation of roughly. Another consistency check is
        to make sure the derived minimum value is not an edge sample, that is to say, we have sampled
        fully across the minimum, and are not approaching it.

        This algorithm is likely slower than the normal line search routine when (f) is fast and
        analytic, but the idea behind this algorithm is to absolutely minimize the number of times
        we have to evaluate (f), and when we do, allows us to do so in batch runs. This could be
        beneficial if evaluating (f) is a very slow.

        A few caveats:
            1. We need to decide how many samples (Nsample) and how far across we should sample
            the search direction. One way to do this is to identify hard parameter space boundaries
            and sample all the way, or a fraction of the way, from the starting point to the
            intersecting boundary edge.

        """
        # Define GP
        GP = gp.GaussianProcessRegressor(gp.kernels.RBF(length_scale=1.0)+gp.kernels.WhiteKernel(1e-4),n_restarts_optimizer=5)

        # Lay Down points
        xS = x0 + np.array(map(lambda x: d*x, np.linspace(0.1,bound,Nsample)))
        
        # Evaluate function
        y = f(xS.T)
        norm = np.abs(y).max()
        y /= norm
        
        # Train GP
        GP.fit(xS, y)
        
        # Get Minimum
        xpred = x0 + np.array(map(lambda x: d*x, np.linspace(0.1,bound,100)))
        ypred = GP.predict(xpred)*norm
        ymin_index = np.where(ypred==ypred.min())[0][0]
        xmin = xpred[ymin_index]
        
        return xmin

    def beta_PR(self, r_i, r_ii):
        """ 
        Polack-Ribiere Beta Factor

        Input:
        ------
        r_i     : ndarray
                Residual vector for step i

        r_ii    : ndarray
                Residual vector for step i + 1

        Output:
        -------
        beta_PR : scalar
                Polack Ribiere factor. If beta_PR < 0 returns 0. 

        """
        beta_PR = np.dot(r_ii.T, (r_ii-r_i))/np.dot(r_i.T,r_i)
        if beta_PR < 0:
            return 0
        else:
            return beta_PR
        
    def descent(self, f, x0, iterations=5, restart=1):
        """
        Perform NLCG descent

        Input:
        ------

        Output:
        -------

        """
        self.pos = []
        d0 = self.norm_gradient(f, x0)
        r0 = d0
        for i in range(iterations):
            self.pos.append(x0)
            x0 = self.GP_line_search(f, x0, d0)
            r1 = self.norm_gradient(f, x0)
            beta = self.beta_PR(r0, r1)
            if i % restart == 0 and i != 0: beta = 0.0
            d0 = r1 + beta*d0
            r0 = r1
        self.pos.append(x0)
        self.pos = np.array(self.pos)
        return x0


















