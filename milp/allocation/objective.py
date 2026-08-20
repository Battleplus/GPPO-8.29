import mip
from allocation.parameters import ParameterBuilder
from allocation.decision_variables import VariableIndices, DecisionVariableSet
from config.settings import GlobalSettings


class ObjectiveBuilder:
    """
    构建 MILP 目标函数:
      max  J = λ_S * J_strike + λ_R * J_recon - λ_T * J_risk - λ_C * J_cost

    J_strike = Σ_g val_g * u_g
    J_recon  = Σ_c ρ_c * q_c
    J_risk   = Σ_{h,g} thr_g * (1 - η_{h,g}) * z_{h,g}
    J_cost   = Σ_{h,w,g} c_w * y_{h,w,g} + M_ξ * Σ_c ξ_c
    """

    def __init__(self, params: ParameterBuilder, idx: VariableIndices,
                 settings: GlobalSettings):
        self.params = params
        self.idx = idx
        self.settings = settings

    def build(self, vars_: DecisionVariableSet) -> mip.LinExpr:
        idx = self.idx
        p = self.params
        s = self.settings

        terms = []

        # -- 侦察收益: +λ_R * Σ_c ρ_c * q_c --
        if idx.N_C > 0:
            terms.append(mip.xsum(
                s.lambda_recon * p.cell_priors[c] * vars_.q[c]
                for c in range(idx.N_C)
            ))

        # -- 距离成本: -λ_d * Σ_{u,s,c} D_uc[u,c] * x_{u,s,c} --
        if idx.N_C > 0:
            terms.append(mip.xsum(
                -s.lambda_distance * p.D_uc[u, c] * vars_.x[(u, si, c)]
                for u in range(idx.N_U)
                for si in range(idx.N_S)
                for c in range(idx.N_C)
            ))

        # -- 打击收益: +λ_S * Σ_g val_g * u_g --
        if idx.N_G > 0 and idx.N_H > 0:
            terms.append(mip.xsum(
                s.lambda_strike * p.target_values[g] * vars_.u[g]
                for g in range(idx.N_G)
            ))

        # -- 风险惩罚: -λ_T * Σ_{h,g} thr_g * (1 - η_{h,g}) * z_{h,g} --
        if idx.N_G > 0 and idx.N_H > 0:
            terms.append(mip.xsum(
                -s.lambda_risk * p.target_threats[g] *
                (1.0 - p.eta_hg[h, g]) * vars_.z[(h, g)]
                for h in range(idx.N_H) for g in range(idx.N_G)
            ))

        # -- 成本惩罚: -λ_C * Σ_{h,w,g} c_w * y_{h,w,g} --
        if idx.N_G > 0 and idx.N_H > 0:
            terms.append(mip.xsum(
                -s.lambda_cost * p.weapon_costs[w] * vars_.y[(h, w, g)]
                for h in range(idx.N_H)
                for w in range(idx.N_W)
                for g in range(idx.N_G)
            ))

        # -- 覆盖松弛惩罚: -λ_C * M_ξ * Σ_c ξ_c --
        if idx.N_C > 0:
            terms.append(mip.xsum(
                -s.lambda_cost * s.big_m_xi * vars_.xi[c]
                for c in range(idx.N_C)
            ))

        if terms:
            return mip.xsum(terms)
        return mip.LinExpr()
