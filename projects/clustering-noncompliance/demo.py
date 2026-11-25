"""Audit clustered uncertainty and the denominators in a non-compliance exercise."""
import json


def effect_summary(estimate, standard_error):
    if standard_error <= 0:
        raise ValueError('Standard error must be positive')
    return {'estimate':estimate, 'standard_error':standard_error,
            'normal_95pct_interval':[estimate-1.96*standard_error, estimate+1.96*standard_error]}


def wald_comparison(outcome_a, uptake_a, outcome_b, uptake_b):
    if not all(0 <= r <= 1 for r in (outcome_a,uptake_a,outcome_b,uptake_b)):
        raise ValueError('Rates must lie between zero and one')
    if uptake_a == uptake_b:
        raise ValueError('A nonzero change in treatment uptake is required')
    return {'intention_to_treat':round(outcome_a-outcome_b,8),
            'change_in_uptake':round(uptake_a-uptake_b,8),
            'wald_ratio':round((outcome_a-outcome_b)/(uptake_a-uptake_b),8)}


def run():
    return {'evidence':'Arithmetic audit of a hypothetical exercise and saved original notebook statistics; source cluster.csv is unavailable',
            'independent_rows':effect_summary(3.60038876098929,.09392981423715299),
            'store_level':effect_summary(3.6003887609905263,2.0873664934199136),
            'email_vs_control':wald_comparison(.66,.4,.64,0),
            'email_call_vs_control':wald_comparison(.68,.6,.64,0),
            'email_call_vs_email':wald_comparison(.68,.6,.66,.4)}


if __name__ == '__main__':print(json.dumps(run(),indent=2))
