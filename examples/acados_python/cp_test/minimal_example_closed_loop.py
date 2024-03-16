# -*- coding: future_fstrings -*-
#
# Copyright (c) The acados authors.
#
# This file is part of acados.
#
# The 2-Clause BSD License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.;
#

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from gamma_model import export_gamma_ode_model
from utils import plot_pendulum
import numpy as np
import scipy.linalg
from casadi import vertcat

def setup(x0, Fmax, N_horizon, Tf):
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # set model
    model = export_gamma_ode_model()
    

    ocp.model = model

    nx = model.x.size()[0] # number of states
    nu = model.u.size()[0] # number of inputs

    ocp.dims.N = N_horizon  # set prediction horizon

    # set cost module
    # set cost
    Q_mat = np.diag([0.0, 1.0]) # weight matrix for state - penalty on gamma_dot
    R_mat = np.diag([1.0]) # weight matrix for control input - u, which directly applied to gamma_dotdot

    # the 'EXTERNAL' cost type can be used to define general cost terms
    # NOTE: This leads to additional (exact) hessian contributions when using GAUSS_NEWTON hessian.
    ocp.cost.cost_type = 'EXTERNAL'
    ocp.cost.cost_type_e = 'EXTERNAL'
    ocp.model.cost_expr_ext_cost = ((model.x.T @ Q_mat @ model.x) - 1)*((model.x.T @ Q_mat @ model.x) - 1) + model.u.T @ R_mat @ model.u # (gamma_dot^2 - 1)^2 + u^2
    ocp.model.cost_expr_ext_cost_e = ((model.x.T @ Q_mat @ model.x) - 1)*((model.x.T @ Q_mat @ model.x) - 1) # (gamma_dot^2 - 1)^2

    # set constraints
    ocp.constraints.x0 = x0 # initial state
    #ocp.constraints.idxbu = np.array([0]) 

    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON' 
    ocp.solver_options.integrator_type = 'IRK'
    ocp.solver_options.sim_method_newton_iter = 10
    ocp.solver_options.nlp_solver_type = 'SQP'
        

    ocp.solver_options.qp_solver_cond_N = N_horizon

    # set prediction horizon
    ocp.solver_options.tf = Tf

    solver_json = 'acados_ocp_' + model.name + '.json'
    acados_ocp_solver = AcadosOcpSolver(ocp, json_file = solver_json)

    # create an integrator with the same settings as used in the OCP solver.
    acados_integrator = AcadosSimSolver(ocp, json_file = solver_json)

    return acados_ocp_solver, acados_integrator


def main():

    x0 = np.array([4.5, 9.5]) # initial state
    Fmax = 50 # used only for plotting the results


    Tf = 12 #.8 # prediction horizon
    N_horizon = 100 # 40 # number of control intervals - 32 [s]

    ocp_solver, integrator = setup(x0, Fmax, N_horizon, Tf) # create ocp solver and integrator

    nx = ocp_solver.acados_ocp.dims.nx # number of states
    nu = ocp_solver.acados_ocp.dims.nu # number of inputs

    Nsim = 100 # 100 # simulation length 80 [s]
    simX = np.ndarray((Nsim+1, nx))
    simU = np.ndarray((Nsim, nu))

    simX[0,:] = x0 # initial state


    t = np.zeros((Nsim))

    # do some initial iterations to start with a good initial guess
    num_iter_initial = 5
    for _ in range(num_iter_initial):
        ocp_solver.solve_for_x0(x0_bar = x0) 

    # closed loop
    for i in range(Nsim):

        # solve ocp and get next control input
        simU[i,:] = ocp_solver.solve_for_x0(x0_bar = simX[i, :])

        t[i] = ocp_solver.get_stats('time_tot')

        # simulate system
        simX[i+1, :] = integrator.simulate(x=simX[i, :], u=simU[i,:])

    # evaluate timings

        # scale to milliseconds
    t *= 1000
    print(f'Computation time in ms: min {np.min(t):.3f} median {np.median(t):.3f} max {np.max(t):.3f}')

    # plot results
    plot_pendulum(np.linspace(0, (Tf/N_horizon)*Nsim, Nsim+1), Fmax, simU, simX)

    ocp_solver = None


if __name__ == '__main__':
    main()
    
