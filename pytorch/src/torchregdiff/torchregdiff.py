import numpy as np
import scipy as sp
import torch

# NOTE - all matrix/vector operations are called using torch.matvel/torch.subtract, etc. Only scalars should multiply ndarrays.


def matrix_diff(n, dx=1.0, dtype=None, device='cpu'):
    """return the matrix representations of the first-order central finite first difference, first-order central second
    finite difference, and simple finite difference for a specified size and delta x increment.
    :param n: the size of the requested matrix
    :param dx: the time step
    :param dtype: the dtype of the data tensor
    :param device: the device for the data tensor

    :return:    diff -- first order central finite first difference matrix of size n with free boundary conditions
                diff2 -- first order central finite second difference matrix of size n with free boundary conditions
                d -- simple finite differnceing matrix (nxn-1)
    """
    if dtype is None:
        dtype=torch.get_default_dtype()
    # first central difference
    mul_step = 1 / (2 * dx)
    c = mul_step * np.ones(n - 1)
    diff = sp.sparse.diags_array([-c, c], offsets=[-1, 1]).toarray()
    diff[0, 0:3] = mul_step * np.asarray([-3, 4, -1])
    diff[-1, -3:] = mul_step * np.asarray([1, -4, 3])

    # second central difference
    mul_step = 1 / (dx**2)
    c = mul_step * np.ones(n)
    diff2 = sp.sparse.diags_array(
        [c[0:-1], -2 * c, c[0:-1]], offsets=[-1, 0, 1]
    ).toarray()
    diff2[0, 0:4] = mul_step * np.asarray([2, -5, 4, -1])
    diff2[-1, -4:] = mul_step * np.asarray([-1, 4, -5, 2])

    # first order differencing matrix
    mul_step = 1 / dx
    c = mul_step * np.ones(n)
    d = sp.sparse.diags_array([-c, c], offsets=[0, 1], shape=[n - 1, n]).toarray()

    diff = torch.from_numpy(diff)
    diff2 = torch.from_numpy(diff2)
    d = torch.from_numpy(d)

    diff = diff.to(copy=True,dtype=dtype,device=device)
    diff2 = diff2.to(copy=True,dtype=dtype,device=device)
    d = d.to(copy=True,dtype=dtype,device=device)

    return diff, diff2, d


def matrix_anti_diff(n, dx=1.0, dtype=None, device='cpu'):
    """return the matrix representation of the antidifferentiation operator for an input of size n
    :param n: the size of the requested matrix
    :param dx: the time step
    :param dtype: the dtype of the data tensor
    :param device: the device for the data tensor

    :return:    a -- cumulative trapezoidal integration matrix that when multiplied by a vector yields the cumulative integral (without the first index) (nxn-1)
                a2 -- cumulative trapezoidal double integration matrix that when multiplied by a vector yields the cumulative double integral (without the first index) (nxn-1)
    """
    if dtype is None:
        dtype=torch.get_default_dtype()
    c = 0.5 * dx * np.ones(n)
    a = np.add(dx * np.tri(n, k=-1), np.diag(c))
    a[:, 0] = 0.5 * dx
    a[0, 0] = 0

    a2 = np.matmul(a, a)

    a = torch.from_numpy(a[1:, :])
    a2 = torch.from_numpy(a2[1:, :])

    a = a.to(copy=True,dtype=dtype,device=device)
    a2 = a2.to(copy=True,dtype=dtype,device=device)

    return a, a2


def reg_diff(
    raw_data: torch.Tensor,
    alpha,
    alpha2=None,
    dx=1.,
    u0=None,
    min_iter=5,
    max_iter=100000,
    delta_cost_tol=1e-6,
    delta_norm_tol=1e-6,
    ep=1e-8,
    skip_first_deriv=False,
    diag=False,
):
    """differentation function that takes a vector of position data and returns the first and second derivatives, evaluated using TV-regularized differentiation.
    :param data: the input data
    :param alpha: the regularization coefficient for the first derivative
    :param alpha2: the regularization coeffiient for the second derivative. If left blank only the first derivative will be calculated.
    :param dx: the time step for the data
    :param u0: an initial guess for the derivative(s) as a 1-D tensor (for the first derivative) or as a 2-D tensor (for both derivatives)
    :param min_iter: minimal number of iterations to calculate before returning. Default value 5.
    :param max_iter: maximum number of iterations to calculate before returning. Default value is 1000000.
    :param delta_cost_tol: relative delta cost to reach iteratively before returning (applies to both derivatives). Default value is 1e-6.
    :param delta_norm_tol: relative delta norm of step to reach iteratively before returning (applies to both derivatives). Default value is 1e-6.
    :param ep: epsilon to use to smooth the absolute value function. Devault value is 1e-8
    :param skip_first_deriv: if True, use the value of u0 for the first derivative and do not calculate it
    :param diag: if True, prints diagnostic information during execution. Default False.

    :return:    u -- the first derivative
                v -- the second derivative
    """
    data = raw_data.squeeze()
    if data.ndim != 1:
        raise ValueError("Expected a data tensor that can be squeezed to 1D")
    n = len(data)
    data_mean = data.mean()
    data_st_dev = data.std(correction=0)
    data = (data - data_mean) / data_st_dev

    if alpha2:
        second_deriv = True
    else:
        second_deriv = False

    diff, diff2, d = matrix_diff(n, dtype=data.dtype, device=data.device)
    d_tp = torch.transpose(d,0,1)

    a, a2 = matrix_anti_diff(n, dtype=data.dtype, device=data.device)
    a_tp = torch.transpose(a,0,1)
    a_tp_a = torch.matmul(a_tp, a)
    a_tp_a2 = torch.matmul(a_tp, a2)

    # bunch of code for handling an input u0
    u = None
    v = None
    if u0 is None and skip_first_deriv:
        raise RuntimeError("If first derivative calculation should be skipped, u0 should be provided")
    if u0 is not None:
        if torch.is_tensor(u0):
            if u0.device != data.device:
                raise ValueError("u0 and data must use the same device")
            if u0.ndim == 1:
                if len(u0) == n:
                    u = u0 * dx / data_st_dev
                else:
                    raise ValueError("Expected a first-derivative u0 tensor that is the same size as the data tensor")
            elif u0.shape[0] == 2 and u0.shape[1] == n:
                    u = u0[0] * dx / data_st_dev
                    v = u0[1] * (dx**2) / data_st_dev
            elif u0.shape[0] == n and u0.shape[1] == 2:
                    u = u0[:,0] * dx / data_st_dev
                    v = u0[:,1] * (dx**2) / data_st_dev
            else:
                raise ValueError("Bad u0 tensor size")
        else:
            raise ValueError("u0 should be a tensor")
    if not torch.is_tensor(u):
        u = torch.mv(diff, data)
    if not torch.is_tensor(v) and second_deriv:
        v = torch.mv(diff2, data)

    # start calculating u
    ofst_1 = data[0]
    if not skip_first_deriv:
        data_comp = data[1:] - ofst_1
        a_tp_b = torch.mv(a_tp, data_comp)
        if diag:
            u_cost = [
                torch.linalg.vector_norm(torch.subtract(torch.cumulative_trapezoid(u), data_comp))
                + alpha * torch.linalg.vector_norm(torch.diff(u), ord=1)
            ]
        prev_cost = None
        for iter in range(max_iter):
            c = torch.reciprocal(torch.sqrt(torch.pow(torch.diff(u), 2) + ep))
            En = torch.diag(c)
            Ln = torch.mm(torch.mm(d_tp, En), d)
            gn = torch.add(torch.subtract(torch.mv(a_tp_a, u), a_tp_b), alpha * torch.mv(Ln, u))
            Hn = torch.add(a_tp_a, alpha * Ln)
            s = torch.linalg.solve(Hn, gn)
            u = torch.subtract(u, s)
            cost_n = (
                torch.linalg.vector_norm(torch.subtract(torch.cumulative_trapezoid(u), data_comp))
                + alpha * torch.linalg.vector_norm(torch.diff(u), ord=1)
            )
            if diag:
                print(
                    f"u, iteration {iter + 1}: cost: {cost_n}, relative change: {torch.linalg.vector_norm(s) / torch.linalg.vector_norm(u)}"
                )
                u_cost.append(cost_n)
            if iter >= min_iter and torch.linalg.vector_norm(s) / torch.linalg.vector_norm(u) < delta_norm_tol:
                if diag:
                    print("u, delta norm tolerance reached")
                break
            if (
                prev_cost
                and iter >= min_iter
                and (prev_cost - cost_n) / prev_cost < delta_cost_tol
            ):
                if diag:
                    print("u, relative cost tolerance reached")
                break
            prev_cost = cost_n

    if second_deriv:
        ofst_2 = u[0]
        data_comp = torch.subtract(data[1:] - ofst_1, ofst_2 * torch.arange(1, n))
        a_tp_b = torch.mv(a_tp, data_comp)
        if diag:
            v_cost = [
                torch.linalg.vector_norm(
                    torch.subtract(
                        torch.cumulative_trapezoid(
                            torch.cat((
                                torch.tensor([0]),
                                torch.cumulative_trapezoid(v)
                            ))
                        ),
                        data_comp,
                    )
                )
                + alpha2 * torch.linalg.vector_norm(torch.diff(v), ord=1)
            ]
        prev_cost = None
        for iter in range(max_iter):
            c = torch.reciprocal(torch.sqrt(torch.pow(torch.diff(v), 2) + ep))
            En = torch.diag(c)
            Ln = torch.mm(torch.mm(d_tp, En), d)
            gn = torch.add(
                torch.subtract(torch.mv(a_tp_a2, v), a_tp_b), alpha2 * torch.mv(Ln, v)
            )
            Hn = torch.add(a_tp_a2, alpha2 * Ln)
            s = torch.linalg.solve(Hn, gn)
            v = torch.subtract(v, s)
            cost_n = (
                torch.linalg.vector_norm(
                    torch.subtract(
                        torch.cumulative_trapezoid(
                            torch.cat((
                                torch.tensor([0]),
                                torch.cumulative_trapezoid(v)
                            ))
                        ),
                        data_comp,
                    )
                )
                + alpha2 * torch.linalg.vector_norm(torch.diff(v), ord=1)
            )
            if diag:
                print(
                    f"v, iteration {iter + 1}: cost: {cost_n}, relative change: {torch.linalg.vector_norm(s) / torch.linalg.vector_norm(v)}"
                )
                v_cost.append(cost_n)
            if iter >= min_iter and torch.linalg.vector_norm(s) / torch.linalg.vector_norm(v) < delta_norm_tol:
                if diag:
                    print("v, delta norm tolerance reached")
                break
            if (
                prev_cost
                and iter >= min_iter
                and (prev_cost - cost_n) / prev_cost < delta_cost_tol
            ):
                if diag:
                    print("v, relative cost tolerance reached")
                break
            prev_cost = cost_n
        v = v * data_st_dev / (dx**2)
    returns = []
    if not skip_first_deriv:
        u = u * data_st_dev / dx
        returns.append(u)
    if second_deriv:
        returns.append(v)
    return (*returns,)