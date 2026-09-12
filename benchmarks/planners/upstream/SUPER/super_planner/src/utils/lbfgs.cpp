#include "utils/optimization/lbfgs.h"

const char *math_utils::lbfgs::lbfgs_strerror(const int err) {
    switch (err) {
        case LBFGS_CONVERGENCE:
            return "Success: reached convergence (g_epsilon).";

        case LBFGS_STOP:
            return "Success: met stopping criteria (past f decrease less than delta).";

        case LBFGS_CANCELED:
            return "The iteration has been canceled by the monitor callback.";

        case LBFGSERR_UNKNOWNERROR:
            return "Unknown error.";

        case LBFGSERR_INVALID_N:
            return "Invalid number of variables specified.";

        case LBFGSERR_INVALID_MEMSIZE:
            return "Invalid parameter lbfgs_parameter_t::mem_size specified.";

        case LBFGSERR_INVALID_GEPSILON:
            return "Invalid parameter lbfgs_parameter_t::g_epsilon specified.";

        case LBFGSERR_INVALID_TESTPERIOD:
            return "Invalid parameter lbfgs_parameter_t::past specified.";

        case LBFGSERR_INVALID_DELTA:
            return "Invalid parameter lbfgs_parameter_t::delta specified.";

        case LBFGSERR_INVALID_MINSTEP:
            return "Invalid parameter lbfgs_parameter_t::min_step specified.";

        case LBFGSERR_INVALID_MAXSTEP:
            return "Invalid parameter lbfgs_parameter_t::max_step specified.";

        case LBFGSERR_INVALID_FDECCOEFF:
            return "Invalid parameter lbfgs_parameter_t::f_dec_coeff specified.";

        case LBFGSERR_INVALID_SCURVCOEFF:
            return "Invalid parameter lbfgs_parameter_t::s_curv_coeff specified.";

        case LBFGSERR_INVALID_MACHINEPREC:
            return "Invalid parameter lbfgs_parameter_t::machine_prec specified.";

        case LBFGSERR_INVALID_MAXLINESEARCH:
            return "Invalid parameter lbfgs_parameter_t::max_linesearch specified.";

        case LBFGSERR_INVALID_FUNCVAL:
            return "The function value became NaN or Inf.";

        case LBFGSERR_MINIMUMSTEP:
            return "The line-search step became smaller than lbfgs_parameter_t::min_step.";

        case LBFGSERR_MAXIMUMSTEP:
            return "The line-search step became larger than lbfgs_parameter_t::max_step.";

        case LBFGSERR_MAXIMUMLINESEARCH:
            return "Line search reaches the maximum try number, assumptions not satisfied or precision not achievable.";

        case LBFGSERR_MAXIMUMITERATION:
            return "The algorithm routine reaches the maximum number of iterations.";

        case LBFGSERR_WIDTHTOOSMALL:
            return "Relative search interval width is at least lbfgs_parameter_t::machine_prec.";

        case LBFGSERR_INVALIDPARAMETERS:
            return "A logic error (negative line-search step) occurred.";

        case LBFGSERR_INCREASEGRADIENT:
            return "The current search direction increases the cost function value.";

        default:
            return "(unknown)";
    }
}

int math_utils::lbfgs::lbfgs_optimize(Eigen::VectorXd &x, double &f,
                                      lbfgs_evaluate_t proc_evaluate,
                                      lbfgs_stepbound_t proc_stepbound,
                                      lbfgs_progress_t proc_progress,
                                      void *instance,
                                      const lbfgs_parameter_t &param) {
    int ret, i, j, k, ls, end, bound;
    double step, step_min, step_max, fx, ys, yy;
    double gnorm_inf, xnorm_inf, beta, rate, cau;

    const int n = x.size();
    const int m = param.mem_size;

    /* Check the input parameters for errors. */
    if (n <= 0) {
        return LBFGSERR_INVALID_N;
    }
    if (m <= 0) {
        return LBFGSERR_INVALID_MEMSIZE;
    }
    if (param.g_epsilon < 0.0) {
        return LBFGSERR_INVALID_GEPSILON;
    }
    if (param.past < 0) {
        return LBFGSERR_INVALID_TESTPERIOD;
    }
    if (param.delta < 0.0) {
        return LBFGSERR_INVALID_DELTA;
    }
    if (param.min_step < 0.0) {
        return LBFGSERR_INVALID_MINSTEP;
    }
    if (param.max_step < param.min_step) {
        return LBFGSERR_INVALID_MAXSTEP;
    }
    if (!(param.f_dec_coeff > 0.0 &&
          param.f_dec_coeff < 1.0)) {
        return LBFGSERR_INVALID_FDECCOEFF;
    }
    if (!(param.s_curv_coeff < 1.0 &&
          param.s_curv_coeff > param.f_dec_coeff)) {
        return LBFGSERR_INVALID_SCURVCOEFF;
    }
    if (!(param.machine_prec > 0.0)) {
        return LBFGSERR_INVALID_MACHINEPREC;
    }
    if (param.max_linesearch <= 0) {
        return LBFGSERR_INVALID_MAXLINESEARCH;
    }

    /* Prepare intermediate variables. */
    Eigen::VectorXd xp(n);
    Eigen::VectorXd g(n);
    Eigen::VectorXd gp(n);
    Eigen::VectorXd d(n);
    Eigen::VectorXd pf(std::max(1, param.past));

    /* Initialize the limited memory. */
    Eigen::VectorXd lm_alpha = Eigen::VectorXd::Zero(m);
    Eigen::MatrixXd lm_s = Eigen::MatrixXd::Zero(n, m);
    Eigen::MatrixXd lm_y = Eigen::MatrixXd::Zero(n, m);
    Eigen::VectorXd lm_ys = Eigen::VectorXd::Zero(m);

    /* Construct a callback data. */
    callback_data_t cd;
    cd.instance = instance;
    cd.proc_evaluate = proc_evaluate;
    cd.proc_stepbound = proc_stepbound;
    cd.proc_progress = proc_progress;

    /* Evaluate the function value and its gradient. */
    fx = cd.proc_evaluate(cd.instance, x, g);

    /* Store the initial value of the cost function. */
    pf(0) = fx;

    /*
    Compute the direction;
    we assume the initial hessian matrix H_0 as the identity matrix.
    */
    d = -g;

    /*
    Make sure that the initial variables are not a stationary point.
    */
    gnorm_inf = g.cwiseAbs().maxCoeff();
    xnorm_inf = x.cwiseAbs().maxCoeff();

    if (gnorm_inf / std::max(1.0, xnorm_inf) <= param.g_epsilon) {
        /* The initial guess is already a stationary point. */
        ret = LBFGS_CONVERGENCE;
    } else {
        /*
        Compute the initial step:
        */
        step = 1.0 / d.norm();

        k = 1;
        end = 0;
        bound = 0;

        while (true) {
            /* Store the current position and gradient vectors. */
            xp = x;
            gp = g;

            /* If the step bound can be provided dynamically, then apply it. */
            step_min = param.min_step;
            step_max = param.max_step;
            if (cd.proc_stepbound) {
                step_max = cd.proc_stepbound(cd.instance, xp, d);
                step_max = step_max < param.max_step ? step_max : param.max_step;
            }
            step = step < step_max ? step : 0.5 * step_max;

            /* Search for an optimal step. */
            ls = line_search_lewisoverton(x, fx, g, step, d, xp, gp, step_min, step_max, cd, param);

            if (ls < 0) {
                /* Revert to the previous point. */
                x = xp;
                g = gp;
                ret = ls;
                break;
            }

            /* Report the progress. */
            if (cd.proc_progress) {
                if (cd.proc_progress(cd.instance, x, g, fx, step, k, ls)) {
                    ret = LBFGS_CANCELED;
                    break;
                }
            }

            /*
            Convergence test.
            The criterion is given by the following formula:
            ||g(x)||_inf / max(1, ||x||_inf) < g_epsilon
            */
            gnorm_inf = g.cwiseAbs().maxCoeff();
            xnorm_inf = x.cwiseAbs().maxCoeff();
            if (gnorm_inf / std::max(1.0, xnorm_inf) < param.g_epsilon) {
                /* Convergence. */
                ret = LBFGS_CONVERGENCE;
                break;
            }

            /*
            Test for stopping criterion.
            The criterion is given by the following formula:
            |f(past_x) - f(x)| / max(1, |f(x)|) < \delta.
            */
            if (0 < param.past) {
                /* We don't test the stopping criterion while k < past. */
                if (param.past <= k) {
                    /* The stopping criterion. */
                    rate = std::fabs(pf(k % param.past) - fx) / std::max(1.0, std::fabs(fx));

                    if (rate < param.delta) {
                        ret = LBFGS_STOP;
                        break;
                    }
                }

                /* Store the current value of the cost function. */
                pf(k % param.past) = fx;
            }

            if (param.max_iterations != 0 && param.max_iterations <= k) {
                /* Maximum number of iterations. */
                ret = LBFGSERR_MAXIMUMITERATION;
                break;
            }

            /* Count the iteration number. */
            ++k;

            /*
            Update vectors s and y:
            s_{k+1} = x_{k+1} - x_{k} = \step * d_{k}.
            y_{k+1} = g_{k+1} - g_{k}.
            */
            lm_s.col(end) = x - xp;
            lm_y.col(end) = g - gp;

            /*
            Compute scalars ys and yy:
            ys = y^t \cdot s = 1 / \rho.
            yy = y^t \cdot y.
            Notice that yy is used for scaling the hessian matrix H_0 (Cholesky factor).
            */
            ys = lm_y.col(end).dot(lm_s.col(end));
            yy = lm_y.col(end).squaredNorm();
            lm_ys(end) = ys;

            /* Compute the negative of gradients. */
            d = -g;

            /*
            Only cautious update is performed here as long as
            (y^t \cdot s) / ||s_{k+1}||^2 > \epsilon * ||g_{k}||^\alpha,
            where \epsilon is the cautious factor and a proposed value
            for \alpha is 1.
            This is not for enforcing the PD of the approxomated Hessian
            since ys > 0 is already ensured by the weak Wolfe condition.
            This is to ensure the global convergence as described in:
            Dong-Hui Li and Masao Fukushima. On the global convergence of
            the BFGS method for nonconvex unconstrained optimization problems.
            SIAM Journal on Optimization, Vol 11, No 4, pp. 1054-1064, 2011.
            */
            cau = lm_s.col(end).squaredNorm() * gp.norm() * param.cautious_factor;

            if (ys > cau) {
                /*
                Recursive formula to compute dir = -(H \cdot g).
                This is described in page 779 of:
                Jorge Nocedal.
                Updating Quasi-Newton Matrices with Limited Storage.
                Mathematics of Computation, Vol. 35, No. 151,
                pp. 773--782, 1980.
                */
                ++bound;
                bound = m < bound ? m : bound;
                end = (end + 1) % m;

                j = end;
                for (i = 0; i < bound; ++i) {
                    j = (j + m - 1) % m; /* if (--j == -1) j = m-1; */
                    /* \alpha_{j} = \rho_{j} s^{t}_{j} \cdot q_{k+1}. */
                    lm_alpha(j) = lm_s.col(j).dot(d) / lm_ys(j);
                    /* q_{i} = q_{i+1} - \alpha_{i} y_{i}. */
                    d += (-lm_alpha(j)) * lm_y.col(j);
                }

                d *= ys / yy;

                for (i = 0; i < bound; ++i) {
                    /* \beta_{j} = \rho_{j} y^t_{j} \cdot \gamm_{i}. */
                    beta = lm_y.col(j).dot(d) / lm_ys(j);
                    /* \gamm_{i+1} = \gamm_{i} + (\alpha_{j} - \beta_{j}) s_{j}. */
                    d += (lm_alpha(j) - beta) * lm_s.col(j);
                    j = (j + 1) % m; /* if (++j == m) j = 0; */
                }
            }

            /* The search direction d is ready. We try step = 1 first. */
            step = 1.0;
        }
    }

    /* Return the final value of the cost function. */
    f = fx;

    return ret;
}

int math_utils::lbfgs::line_search_lewisoverton(Eigen::VectorXd &x, double &f, Eigen::VectorXd &g, double &stp,
                                                const Eigen::VectorXd &s, const Eigen::VectorXd &xp,
                                                const Eigen::VectorXd &gp, const double stpmin, const double stpmax,
                                                const callback_data_t &cd, const lbfgs_parameter_t &param) {
    int count = 0;
    bool brackt = false, touched = false;
    double finit, dginit, dgtest, dstest;
    double mu = 0.0, nu = stpmax;

    /* Check the input parameters for errors. */
    if (!(stp > 0.0)) {
        return LBFGSERR_INVALIDPARAMETERS;
    }

    /* Compute the initial gradient in the search direction. */
    dginit = gp.dot(s);

    /* Make sure that s points to a descent direction. */
    if (0.0 < dginit) {
        return LBFGSERR_INCREASEGRADIENT;
    }

    /* The initial value of the cost function. */
    finit = f;
    dgtest = param.f_dec_coeff * dginit;
    dstest = param.s_curv_coeff * dginit;

    while (true) {
        x = xp + stp * s;

        /* Evaluate the function and gradient values. */
        f = cd.proc_evaluate(cd.instance, x, g);
        ++count;

        /* Test for errors. */
        if (std::isinf(f) || std::isnan(f)) {
            return LBFGSERR_INVALID_FUNCVAL;
        }
        /* Check the Armijo condition. */
        if (f > finit + stp * dgtest) {
            nu = stp;
            brackt = true;
        } else {
            /* Check the weak Wolfe condition. */
            if (g.dot(s) < dstest) {
                mu = stp;
            } else {
                return count;
            }
        }
        if (param.max_linesearch <= count) {
            /* Maximum number of iteration. */
            return LBFGSERR_MAXIMUMLINESEARCH;
        }
        if (brackt && (nu - mu) < param.machine_prec * nu) {
            /* Relative interval width is at least machine_prec. */
            return LBFGSERR_WIDTHTOOSMALL;
        }

        if (brackt) {
            stp = 0.5 * (mu + nu);
        } else {
            stp *= 2.0;
        }

        if (stp < stpmin) {
            /* The step is the minimum value. */
            return LBFGSERR_MINIMUMSTEP;
        }
        if (stp > stpmax) {
            if (touched) {
                /* The step is the maximum value. */
                return LBFGSERR_MAXIMUMSTEP;
            } else {
                /* The maximum value should be tried once. */
                touched = true;
                stp = stpmax;
            }
        }
    }
}