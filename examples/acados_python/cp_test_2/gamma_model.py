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

from acados_template import AcadosModel
from casadi import SX, vertcat, sin, cos

def export_gamma_ode_model() -> AcadosModel:

    model_name = 'gamma_ode'

    # constants

    # set up states & controls
    x1      = SX.sym('x1') # gamma
    x2      = SX.sym('x2') # gamma_dot
    x3      = SX.sym('x3') # L_12 = L_21
    x4      = SX.sym('x4') # gamma_2
    x5      = SX.sym('x5') # G_11
    x6      = SX.sym('x6') # G_11

    x = vertcat(x1, x2, x3, x4, x5, x6)

    U = SX.sym('U') # gamma_dot_dot - direct control over the acceleration
    u = vertcat(U)

    x1_dot      = SX.sym('x1_dot')
    x2_dot      = SX.sym('x2_dot')
    x3_dot      = SX.sym('x3_dot')
    x4_dot      = SX.sym('x4_dot')
    x5_dot      = SX.sym('x5_dot')
    x6_dot      = SX.sym('x6_dot')

    xdot = vertcat(x1_dot, x2_dot, x3_dot, x4_dot, x5_dot, x6_dot)

    # dynamics
    # Explicit Runge-Kutta 4 integrator (erk) - dot{x} = f(x, u ,p)
    f_expl = vertcat(x2, U, 0, 0, 0, 0) # dx1 = x2, dx2 = U
                     

    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = model_name

    return model

