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

    def calc_jacobian(self, f, theta, diff_vec, attach=True, grad_type='forward', f_args=[]):
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
        self.f0 = f(theta, *f_args)
        pos_vec, neg_vec = self.calc_partials(f, theta, diff_vec, second_order=False, f_args=f_args)

        # Construct Jacobian Matrix
        J = np.empty((1,ndim))
        for i in range(ndim):
            if grad_type == 'forward':
                J[0,i] = self.first_forward(self.f0, pos_vec[i], diff_vec[i])
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

    def calc_partials(self, f, theta, diff_vec, second_order=True, f_args=[]):
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
                f_pos       = f(theta_pos, *f_args)
                f_neg       = f(theta_neg, *f_args)
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

    def find_root(self, f, theta, diff_vec, nsteps=10, gamma=0.1, second_order=False, bounds=None, step_size=None, f_args=[]):
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
                pos_mat, neg_mat = self.calc_partials(f, theta, diff_vec, second_order=second_order, f_args=f_args)
                if second_order == True:
                    H, J = self.calc_hessian(f(theta, *f_args), pos_mat, neg_mat, diff_vec)
                    grads.append([self.H, self.J])
                else:
                    J = self.calc_jacobian(f(theta, *f_args), pos_mat, diff_vec, neg_vec=neg_mat, f_args=f_args)
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

    def hess(self, theta, f_args=[]):
        pos_mat, neg_mat = self.calc_partials(self.f, theta, self.diff_vec, second_order=True, f_args=f_args)
        H, J = self.calc_hessian(self.f(theta, *f_args), pos_mat, neg_mat, self.diff_vec)
        return H

    def jac(self, theta, f_args=[]):
        pos_vec, neg_vec = self.calc_partials(self.f, theta, self.diff_vec, second_order=False, f_args=f_args)
        J = self.calc_jacobian(self.f, pos_vec, self.diff_vec, neg_vec=neg_vec, f_args=f_args)
        return J[0]


class Conj_Grad(FiniteDiff):

    def __init__(self):
        """
        Exposed routines for the Nonlinear Conjugate Gradient algorithm,
        designed for when the underlying minimizable function (f) is expensive
        to evaluate and non-analytic (e.g. a complex computer simulation). 

        In order to use it with a computer simulation, one must construct 
        their own function (f) that submits job, writes data to file, loads
        it in and returns relevant data.
        """
        pass

    def print_message(self, msg, type=1):
        if type == 0:
            print(msg)
        if type == 1:
            print('\n'+msg+'\n'+'-'*30)

    def weave(self, a, b):
        """
        weave two ndarrays 'a' and 'b' together
        """
        ashape = a.shape
        bshape = b.shape
        if len(ashape) == 1:
            cshape = (ashape[0] + bshape[0],)
        else:
            cshape = (ashape[0]+bshape[0], ashape[1])
        c = np.empty(cshape,dtype=a.dtype)
        c[0::2] = a
        c[1::2] = b
        return c

    def initialize(self, base_dir):
        """
        Initialize the communications between this script and the outside world

        Input:
        ------


        """
        pass

    def norm_gradient(self, f, x0, dx=0.01, grad_type='forward', f_args=[]):
        """
        Negative unit vector of gradient

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
                Negative gradient unit vector
        """
        length = len(x0)
        if type(dx) != np.ndarray:
            dx = np.ones(length)*dx

        grad = -self.calc_jacobian(f, x0, dx, grad_type=grad_type, attach=False, f_args=f_args)[0]
        grad /= la.norm(grad)
        return grad
        
    def GP_line_search(self, f, x0, y0, d, Nsample=50, Nmin=500, distance_frac=0.5,
                            param_bounds=None, n_restart=10, dist=None, verbose=True,
                            backpace=1, f_args=[]):
        """
        Gaussian Process Line Search

        Input:
        -------



        Output:
        -------
        xmin

        Notes:
        ------
        The idea behind this algorithm is to sample a few points from (f) along the search direction (slow),
        fit their response to a 1D GP (fast), and then solve for its minimum via direct draws from the
        GP mean function along the search direction (fast). To ensure we have not undersampled (f) we lay
        down another set of points in between the original set (slow), fit another GP and compare the 
        the scale length hyperparameter, ell, to the original GP ell (fast). We repeat until ell_2 is roughly
        the same as ell_1. Another stop point could be to stop when the derived minima are
        roughly the same. Both require an interpretation of roughly. Another consistency check is
        to make sure the derived minimum value is not an edge sample, that is to say, we have sampled
        fully across the minimum, and are not approaching it.

        This algorithm is likely slower than the normal line search routine when (f) is fast and
        analytic, but the idea behind this algorithm is to absolutely minimize the number of times
        we have to evaluate (f), and when we do, allow us to do so in batch runs. This could be
        beneficial if evaluating (f) is a very slow and could benefit from a computing cluster.

        A few caveats:
            1. When doing a line search, we need to decide how many samples (Nsample) and how far along the 
            line we should sample. One way to do this is to identify hard parameter space boundaries
            and sample all the way, or a fraction of the way, from the starting point to the
            intersecting boundary edge.

        """
        if verbose == True:
            self.print_message('starting GP line search')

        # Define GP
        GP = gp.GaussianProcessRegressor(gp.kernels.RBF(length_scale=1.0)+gp.kernels.WhiteKernel(1e-5),
                                            n_restarts_optimizer=n_restart)

        # Make dist equal to hard parameter bounds if not given
        if dist is None:
            if param_bounds is None:
                dist = 3.0
            else:
                # Distance to boundary across each parameter
                param_edge = np.array(map(lambda x: x[0][1 if x[1] > 0 else 0], zip(param_bounds, d)))
                delta = param_edge - x0
                # Ratio on grad unit vector
                R = delta / d
                Rmin = np.where(R==R.min())[0][0]
                if type(Rmin) == np.ndarray:
                    Rmin = Rmin[0]
                # Magnitude of vector to param_bound
                dist = R[Rmin]

        # Lay down search points
        xNEW = np.linspace(0,dist*distance_frac,Nsample+1)
        dx = xNEW[1] - xNEW[0]

        # Adjust according to backpace
        xNEW -= dx*backpace

        # Delete starting point
        xL = np.delete(xNEW, backpace, axis=0)
        xS = np.array(map(lambda x: x0 + x*d, xL))
        
        # Evaluate function
        yS = f(xS.T, *f_args)
        
        # Concatenate Arrays
        xS = np.insert(xS, backpace, x0, axis=0)
        xL = np.insert(xL, backpace, 0, axis=0)
        yS = np.insert(yS, backpace, y0, axis=0)

        # Normalize yS
        norm = np.abs(yS).max()
        yS /= norm

        j = 0
        while True:
            # Train GP1 with half of points
            GP.fit(xL[::2][:,np.newaxis], yS[::2])
            
            ## Get Minimum1 by one-factor refining
            # First iteration minima
            xpred = np.linspace(xL[0], xL[-1], Nmin)
            ypred1 = GP.predict(xpred[:,np.newaxis])*norm
            ymin_index1 = np.where(ypred1==ypred1.min())[0][0]
            # new starting and ending points that are 20 elements long around minima
            xpred_start = xpred[ymin_index1-10]
            xpred_dist = xpred[ymin_index1+10]-xpred_start
            # Get minima of refined grid
            xpred = xpred_start + np.linspace(0, xpred_dist, Nmin)
            ypred1 = GP.predict(xpred[:,np.newaxis])*norm
            ymin_index1 = np.where(ypred1==ypred1.min())[0][0]
            dmin1 = xpred[ymin_index1]
            xmin1 = x0 + dmin1*d
            ell1 = np.exp(GP.kernel_.theta[0])
            
            # Train GP2 with all points
            GP.fit(xL[:,np.newaxis], yS)
            
            # Get Minimum2 by one-factor refining
            xpred = np.linspace(xL[0], xL[-1], Nmin)
            ypred2 = GP.predict(xpred[:,np.newaxis])*norm
            ymin_index2 = np.where(ypred2==ypred2.min())[0][0]
            xpred_start = xpred[ymin_index2-10]
            xpred_dist = xpred[ymin_index2+10]-xpred_start
            xpred = xpred_start + np.linspace(0, xpred_dist, Nmin)
            ypred2 = GP.predict(xpred[:,np.newaxis])*norm
            ymin_index2 = np.where(ypred2==ypred2.min())[0][0]
            dmin2 = xpred[ymin_index2]
            xmin2 = x0 + dmin2*d
            ell2 = np.exp(GP.kernel_.theta[0])

            if verbose == True:
                self.print_message('ell_GP1 = %.2f, ell_GP2 = %.2f'%(ell1,ell2),type=0)
                self.print_message('xmin_GP1 = %.2f, xmin_GP2 = %.2f'%(dmin1,dmin2),type=0)

            # Check Minimum is not edge point
            if la.norm(xmin2-xL[-1])/(dist*distance_frac) < 0.01:
                extend_points = True
                if verbose == True:
                    self.print_message('xmin_GP2 is an edge point...', type=0)
            else:
                extend_points = False

            # Finish if the two give similar answers
            if (np.abs(ell2-ell1)/ell1 < 0.5 or np.abs(dmin2-dmin1)/(dist*distance_frac) < 0.1) and extend_points == False:
                break

            # If edge point then extend points, else double down sampling density
            if extend_points == True:
                xL2 = xL[-1] + np.linspace(0,dist*distance_frac,Nsample+1)[1:]
                xS2 = np.array(map(lambda x: x0 + x*d, xL2))

                # Evaluate function
                yS2 = f(xS2.T, *f_args)
                yS2 /= norm

                # Concatenate Arrays
                xL = np.concatenate([xL, xL2])
                xS = np.concatenate([xS, xS2])
                yS = np.concatenate([yS, yS2])

            else:
                dxL = xL[1] - xL[0]
                xL2 = (xL + dxL/2.0)[:-1]
                xS2 = np.array(map(lambda x: x + x*d, xL2))

                # Evaluate function
                yS2 = f(xS2.T, *f_args)
                yS2 /= norm
                
                # Concatenate Arrays
                xS = self.weave(xS, xS2)
                xL = self.weave(xL, xL2)
                yS = self.weave(yS, yS2)

            yS *= norm
            norm = np.abs(yS).max()
            yS /= norm

            xL = np.array(map(lambda x: la.norm(x-x0), xS))

            j += 1
            if verbose == True:
                self.print_message('finished '+str(j+1)+' line searches')

        if verbose == True:
            self.print_message('finished GP line search with xmin = '+str(xmin2))

        return xmin2


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
        
    def GP_descent(self, f, x0, iterations=5, restart=1, grad_kwargs={}, GP_ls_kwargs={}, f_args=[], verbose=True):
        """
        Perform NLCG descent with Gaussian Process line search

        Input:
        ------

        f : function object


        x0 : ndarray


        iterations : int

        restart : int

        GP_Nsample : int
            Number of samples to evaluate along GP line search.
            If you can run K samples in series on your cluster,
            it is recommended to make GP_Nsample = K, and ideally
            is an even number. 


        Output:
        -------
        x0 : ndarray
            parameter vector of stopping point

        """
        self.pos = [x0]
        d0 = self.norm_gradient(f, x0, f_args=f_args, **grad_kwargs)
        r0 = d0
        for i in range(iterations):
            x1 = self.GP_line_search(f, x0, self.f0, d0, f_args=f_args, **GP_ls_kwargs)
            r1 = self.norm_gradient(f, x1, f_args=f_args, **grad_kwargs)
            beta = self.beta_PR(r0, r1)
            if i % restart == 0 and i != 0: beta = 0.0
            d0 = r1 + beta*d0
            r0 = r1
            self.pos.append(np.copy(x1))
            x0 = x1
        self.pos = np.array(self.pos)
        return x0






