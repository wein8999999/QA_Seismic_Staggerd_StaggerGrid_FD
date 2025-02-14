# !/usr/bin/env python
# -*-coding:utf-8 -*-
# @Time    : 2023/6/8 16:19
# @Author  : Wen Xixiang
# @Version : python3.9
import warnings
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

pi = np.pi
fact = np.math.factorial


class Stagger_FD_coeff_1D:
    """
    This is a class that can construct the linear system of equations
    proposed by Wang in his 2014 paper (doi: 10.1190/geo2014-0078.1).
    By solving this linear system of equations, one can obtain the
    optimal finite difference coefficients for the staggered grid finite
    difference method.

    This class is limited to constructing one-dimensional problems.

    Parameters
    ----------
    v : int or float
        The velocity of acoustic waves.
    h : int or float
        The spatial grid spacing.
    tao : int or float
        The temporal grid spacing.
    freq_max : int or float, optional
        This parameter denotes the upper limit of the frequency spectrum
        for the sound wave. If not explicitly specified, it will be computed
        as 90% of the Nyquist frequency.
    kh_total : int or float, optional
        The parameter represents the maximum value obtained by multiplying
        the wavenumber by the spacing of the spatial grid. When the frequency
        reaches the Nyquist frequency, its value is equal to \pi.
    L : int, optional
        The total number of wavenumber points(default=1000).
    """

    def __init__(self, v, h, tao, freq_max=None, kh_total=pi, L=100, *args, **kwargs):
        self._v = v
        self._h = h
        self._tao = tao
        self._r = v * tao / h
        self._L = L  # total number of  wavenumber points
        self.nyquist_freq = self._v / (2 * self._h)
        self.max_frequency = freq_max
        self._kh_total = kh_total
        self._kh_total_vec = np.linspace(
            self._kh_total / self._L, self._kh_total, self._L
        )
        self._N = int(
            np.floor(self._L * self._ratio)
        )  # optimal number of  wavenumber points
        self._kh_vec = self._kh_total_vec[0 : self._N]
        self.kh_optimal = self._kh_vec[-1]

    @property
    def max_frequency(self):
        return self._f

    @max_frequency.setter
    def max_frequency(self, new_max_freq):
        if new_max_freq is None:
            warnings.warn(
                "Max frequency of acoustic wave is not provided! Will take"
                " 90% Nyquist frequency."
            )
            new_max_freq = 0.9 * self.nyquist_freq
            self._f = new_max_freq
            self._ratio = 1
        elif new_max_freq / self.nyquist_freq > 1:
            warnings.warn(
                "The ratio of K_opti/K_total larger than 1, which means the maximum of frequency is "
                "larger than Nyquist sampling frequency! Will take the 90% Nyquist frequency!"
            )
            new_max_freq = 0.9 * self.nyquist_freq
            self._f = new_max_freq
            self._ratio = 1
        else:
            self._f = new_max_freq
            self._ratio = new_max_freq / self.nyquist_freq

    @property
    def total_wavenumber_points(self):
        return self._L

    @property
    def optimal_wavenumber_points(self):
        return self._N

    @property
    def kh_optimal(self):
        return self._kh_optimal

    @kh_optimal.setter
    def kh_optimal(self, new_optimal_kh):
        if not isinstance(new_optimal_kh, (int, float)):
            raise ValueError
        self._kh_optimal = new_optimal_kh

    @property
    def kh_vec(self):
        return self._kh_vec

    def sum_coeff(self, coeff_vec, kh):
        return sum(
            [coeff_vec[i] * np.sin((i + 0.5) * kh) for i in range(len(coeff_vec))]
        )

    def F(self, coeff_vec, kh):
        """Compute Equation 8 in the reference paper."""
        return self.sum_coeff(coeff_vec, kh) ** 2

    def F_partial(self, coeff_vec, kh, index):
        """Compute Equation 14 in the reference paper."""
        return 2 * self.sum_coeff(coeff_vec, kh) * np.sin((index + 0.5) * kh)

    def RHS_d(self, k):
        """Compute Equation 9 in the reference paper."""
        return (np.sin(0.5 * k * self._v * self._tao) / self._r) ** 2

    def construct(self, M, coeff_vec=None):
        """Construct the system of linear equations."""
        if coeff_vec is None:
            coeff_vec = np.ones(M).reshape(-1, 1) / 100
        A = np.zeros([self._N, M])
        b = np.zeros([self._N, 1]).reshape(-1, 1)
        for i in range(self._N):
            b[i] = self.RHS_d(self._kh_vec[i] / self._h) - self.F(
                coeff_vec, self._kh_vec[i]
            )
            for j in range(M):
                A[i, j] = self.F_partial(coeff_vec, self._kh_vec[i], j)
        return A, b

    def construct_lstsq(self, M, coeff_vec=None):
        """Construct the system of linear equations based on the
        least squares method.
        """
        if coeff_vec is None:
            coeff_vec = np.ones(M).reshape(-1, 1) / 100
        A, b = self.construct(M, coeff_vec)
        C = A.T @ A
        c = A.T @ b
        return C, c

    def construct_lstsq_reg(self, M, alpha, coeff_vec=None):
        """Construct the system of linear equations based on the
        least squares method, incorporating regularization techniques.
        """
        if coeff_vec is None:
            coeff_vec = np.ones(M).reshape(-1, 1) / 100
        C, c = self.construct_lstsq(M, coeff_vec)
        D = np.identity(M)
        C_alpha = C + alpha * D
        c_alpha = c - alpha * D @ coeff_vec
        return C_alpha, c_alpha

    def von_Neumann_stab_cond(self, coeff_vec, Dim=1):
        s = 1 / Dim ** (1 / 2) * 1 / (np.sum(np.abs(coeff_vec)))
        if self._r >= s:
            raise ValueError(
                f"r={self._r} >= s={s} which not satisfy the stability condition! "
            )

    def cal_Phi(self, coeff_vec):
        """Calculating the non-linear objective function."""
        return sum(
            [
                (self.F(coeff_vec, kh) - self.RHS_d(kh / self._h)) ** 2
                for kh in self._kh_vec
            ]
        ).squeeze()

    def cal_sigma(self, r, kh_total, coeff_vec):
        """Compute Equation 23 in the reference paper."""
        return 2 / (r * kh_total) * np.arcsin(r * self.sum_coeff(coeff_vec, kh_total))

    def show_dispersion_curve(self, coeff_vec, iter_num):
        fig, ax = plt.subplots(figsize=(4, 4))
        sigma = self.cal_sigma(self._r, self._kh_total_vec, coeff_vec)
        ax.set_title(f"Dispersion errors (iter_num={iter_num})")
        ax.plot(self._kh_total_vec, sigma)
        ax.set_ylim([0.9, 1.05])
        ax.set_xlim([0, 3])
        plt.pause(1)
        plt.close(fig)

    def __cal_lstsq_coeff_delta_vec(self, M, coeff_vec, alpha, solver):
        """Solving the linear system of equation constructed from the least
        square method.
        """
        if alpha == 0:
            C, c = self.construct_lstsq(M, coeff_vec)
            try:
                coeff_delta = solver(C, c)
            except:
                coeff_delta = solver(C + 1.0e-10 * np.eye(M), c)

        elif alpha > 0:
            A_alpha, b = self.construct_lstsq_reg(M, alpha, coeff_vec)
            coeff_delta = solver(A_alpha, b)
        else:
            raise ValueError("Regularization factor alpha should > 0")

        return coeff_delta

    def __cal_normal_coeff_delta_vec(
        self, M, coeff_vec, alpha, solver, solver_param_dict
    ):
        """Solving the linear system of equation."""
        A, b = self.construct(M, coeff_vec)
        coeff_delta = solver(A, b, alpha=alpha, **solver_param_dict)

        return coeff_delta

    def solve(
        self,
        M,
        coeff_vec=None,
        alpha=None,
        beta=1.0,
        beta_decay=1.0,
        max_iter_num=100,
        solver=np.linalg.solve,
        mode_of_construct="lstsq",  # 'normal'
        epsilon=1e-6,
        alpha_decay=1e-1,
        show_process=False,
        add_non_linear_obj=False,
        SampleSolver_param_dict={},
        return_iter_num=False,
        call_back=None,
    ):
        """Solve the system of linear equations constructed based on the
        time-space domain frequency preservation method, yielding the
        optimal finite difference stencil for the staggered grid finite
        difference method.

        Parameters
        ----------
        M : int
            The length of stagger grid finite difference operator.
        coeff_vec : array_like, optional
            The initial vector of coefficients.
        alpha : float or int, optional
            The factor of regularization.
        beta : float or int, optional
            The step length to adjust the speed of iterations.
        max_iter_num : int
            The maximum number of iterations.
        solver : Any
            The tool for solving linear systems of equations.
        mode_of_construct : {'lstsq', 'normal'}, optional
            Default is 'lstsq', which construct the linear system of equations
            by the least square method. The 'normal' mode implies that
            the least squares method is not used when constructing the
            system of linear equations. Such a system of equations is often
            overdetermined.
        epsilon : float
            The upper limit of the solution error.
        show_process : bool
            Whether to show the process of  iterationss.
        add_non_linear_obj : bool
            When using D-Wave Sampler to solving the linear system of equations,
            whether to add the non-linear objective function. This operation
            may slightly improve the effectiveness of the solution but will
            increase the computational time.

        Returns
        -------
        coeff_vec : array_like
            The optimal coefficient vector of staggered grid finite difference.
        """
        ini_alpha = np.copy(alpha)

        if coeff_vec is None:
            coeff_vec = np.ones(M).reshape(-1, 1) / 100

        if len(coeff_vec.shape) == 1 or coeff_vec.shape[1] != 1:
            coeff_vec = coeff_vec.reshape(-1, 1)

        if alpha is None:
            alpha = 0

        for iter_num in tqdm(range(1, max_iter_num + 1)):

            # von Neumann stability condition 1D
            # self.von_Neumann_stab_cond(coeff_vec)

            # show dispersion curve
            if show_process:
                interval = (max_iter_num // 20) if max_iter_num >= 20 else 1
                if iter_num % interval == 0:
                    self.show_dispersion_curve(coeff_vec, iter_num)

            # calculate the object function (eq 10)
            obj = self.cal_Phi(coeff_vec)
            print(f"Current object function value is {obj}")
            print("\n")

            # Optimal sampling result
            if add_non_linear_obj:

                def non_lin_obj(delta_a, current_coeff_vec=coeff_vec):
                    return self.cal_Phi(delta_a + current_coeff_vec)

                SampleSolver_param_dict["non_linear_obj"] = non_lin_obj

            if obj < epsilon:
                print(f"The target accuracy is satisfied and the iteration is ended!")
                break

            if call_back is not None:
                call_back(coeff_vec, obj)

            print(f"~~~~~~~~ Start {iter_num} iteration ~~~~~~~~")

            if mode_of_construct == "lstsq":
                coeff_delta = self.__cal_lstsq_coeff_delta_vec(
                    M, coeff_vec, alpha, solver
                )
            elif mode_of_construct == "direct":
                coeff_delta = self.__cal_normal_coeff_delta_vec(
                    M, coeff_vec, alpha, solver, SampleSolver_param_dict
                )
            else:
                raise NotImplementedError

            alpha = ini_alpha * alpha_decay**iter_num if alpha > 0 else 0

            coeff_vector_new = coeff_vec + beta * coeff_delta
            beta *= beta_decay

            print("Current coeff_vec is:")
            print(f"{coeff_vector_new.reshape(1,-1)}")

            if np.linalg.norm((coeff_vec - coeff_vector_new) / coeff_vec) < epsilon:
                print("Iteration stalled! Stop it.")
                break
            else:
                coeff_vec = coeff_vector_new

        # Don't forget calculating the last objection
        # obj = self.cal_Phi(coeff_vec)
        print(f"Current object function value is {obj}")
        if return_iter_num:
            return coeff_vec, iter_num
        else:
            return coeff_vec

    def _calCond(self, M, coeff_vec):
        """Calculating the condition number of linear system of equations."""
        C, _ = self.construct_lstsq(M, coeff_vec)
        return np.linalg.cond(C)


class Stagger_FD_coeff_2D(Stagger_FD_coeff_1D):
    """
    This class is limited to constructing two-dimensional problems.

    Parameters
    ==========
    max_theta : float or int
        Maximum of theta for solving finite difference operator (Default is \pi).
    num_theta : int
        Number of different direction. (Default is 20)
    """

    def __init__(self, *args, max_theta=2 * pi, num_theta=20, **kwargs):
        super().__init__(*args, **kwargs)
        # self.theta = np.linspace(0, 2*pi, 20)
        self.theta = np.linspace(0, max_theta, num_theta)

    def sum_coeff(self, coeff_vec, kh):
        return sum(
            [coeff_vec[i] * np.sin((i + 0.5) * kh) for i in range(len(coeff_vec))]
        )

    def F(self, coeff_vec, kh):
        return sum(
            [
                self.sum_coeff(coeff_vec, kh * np.cos(theta)) ** 2
                + self.sum_coeff(coeff_vec, kh * np.sin(theta)) ** 2
                for theta in self.theta
            ]
        )

    def F_partial(self, coeff_vec, kh, index):
        return sum(
            [
                2
                * self.sum_coeff(coeff_vec, kh * np.cos(theta))
                * np.sin((index + 0.5) * kh * np.cos(theta))
                + 2
                * self.sum_coeff(coeff_vec, kh * np.sin(theta))
                * np.sin((index + 0.5) * kh * np.sin(theta))
                for theta in self.theta
            ]
        )

    def RHS_d(self, k):
        # return sum([(np.sin(0.5 * k * self._v * self._tao) / self._r) ** 2 for _ in self.theta])
        return (np.sin(0.5 * k * self._v * self._tao) / self._r) ** 2 * len(self.theta)

    def cal_sigma(self, r, kh_total, coeff_vec, theta=pi / 8):
        return (
            2
            / (r * kh_total)
            * np.arcsin(
                r
                * np.sqrt(
                    self.sum_coeff(coeff_vec, kh_total * np.cos(theta)) ** 2
                    + self.sum_coeff(coeff_vec, kh_total * np.sin(theta)) ** 2
                )
            )
        )


class Stagger_FD_coeff_3D(Stagger_FD_coeff_2D):
    """
    This class is limited to constructing three-dimensional problems.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.phi = np.linspace(0, 2 * pi, 20)

    def sum_coeff(self, coeff_vec, kh):
        return sum(
            [coeff_vec[i] * np.sin((i + 0.5) * kh) for i in range(len(coeff_vec))]
        )

    def F(self, coeff_vec, kh):
        return sum(
            [
                self.sum_coeff(coeff_vec, kh * np.cos(theta) * np.cos(phi)) ** 2
                + self.sum_coeff(coeff_vec, kh * np.cos(theta) * np.sin(phi)) ** 2
                + self.sum_coeff(coeff_vec, kh * np.sin(theta)) ** 2
                for theta, phi in zip(self.theta, self.phi)
            ]
        )

    def F_partial(self, coeff_vec, kh, index):
        return sum(
            [
                2
                * self.sum_coeff(coeff_vec, kh * np.cos(theta) * np.cos(phi))
                * np.sin((index + 0.5) * kh * np.cos(theta) * np.cos(phi))
                + 2
                * self.sum_coeff(coeff_vec, kh * np.cos(theta) * np.sin(phi))
                * np.sin((index + 0.5) * kh * np.cos(theta) * np.sin(phi))
                + 2
                * self.sum_coeff(coeff_vec, kh * np.sin(theta))
                * np.sin((index + 0.5) * kh * np.sin(theta))
                for theta, phi in zip(self.theta, self.phi)
            ]
        )

    def RHS_d(self, k):
        return sum(
            [(np.sin(0.5 * k * self._v * self._tao) / self._r) ** 2 for _ in self.theta]
        )

    def cal_sigma(self, r, kh_total, coeff_vec, theta=pi / 8, phi=pi / 4):
        return (
            2
            / (r * kh_total)
            * np.arcsin(
                r
                * np.sqrt(
                    self.sum_coeff(coeff_vec, kh_total * np.cos(theta) * np.cos(phi))
                    ** 2
                    + self.sum_coeff(coeff_vec, kh_total * np.cos(theta) * np.sin(phi))
                    ** 2
                    + self.sum_coeff(coeff_vec, kh_total * np.sin(theta)) ** 2
                )
            )
        )


def Taylor_Coef(M=4):
    """
    Use the traditional Taylor expansion to compute the coefficients of
    the finite difference. I'm a fool...

    Parameters
    ----------
    M : int
        The length of stagger grid finite difference operator.

    Returns
    -------
    coeff_vec : array_like
        The coefficient vector of staggered grid finite difference.
    """

    A = np.zeros([M, M])
    b = np.zeros([M, 1])
    for i in range(M):
        b[i] = 1 if i == 0 else 0
        for j in range(M):
            A[i, j] = (2 * j + 1) ** (2 * i + 1)

    lu, piv = sp.linalg.lu_factor(A)
    x = sp.linalg.lu_solve((lu, piv), b)
    # x = np.linalg.solve(A, b).reshape(-1, 1)
    return x


if __name__ == "__main__":
    v = 1500
    h = 10
    tao = 0.001
    M = 4
    freq = 45

    # linear_sys = Stagger_FD_coeff_1D(v, h, tao, freq)  # test for 1D
    # linear_sys = Stagger_FD_coeff_2D(v, h, tao, freq)  # test for 2D
    linear_sys = Stagger_FD_coeff_3D(v, h, tao, freq)  # test for 2D

    a0 = linear_sys.solve(
        M=M,
        coeff_vec=Taylor_Coef(M),  # np.ones(M).reshape(-1, 1) / 10
        alpha=1,  # 1, 3e-6, 0
        beta=0.7,  # 1,  0.7, # 0.07
        max_iter_num=20,
        solver=np.linalg.solve,
        mode_of_construct="direct",  # 'lstsq', 'direct'
        show_process=False,
    )

    kh_vec = np.linspace(pi / 100, pi, 100)
    taylor_sigma = linear_sys.cal_sigma(
        r=v * tao / h,
        kh_total=kh_vec,
        coeff_vec=Taylor_Coef(M),
    )
    sigma = linear_sys.cal_sigma(
        r=v * tao / h,
        kh_total=kh_vec,
        coeff_vec=a0,
    )

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_title(f"Dispersion errors Taylor")
    ax.plot(kh_vec, taylor_sigma, label="taylor")
    ax.plot(kh_vec, sigma, label="Wang")
    ax.set_ylim([0.9, 1.05])
    ax.set_xlim([0, 3])
    ax.legend()
    plt.show()
