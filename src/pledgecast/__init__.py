"""PledgeCast - explainable ML early-warning for promoter-pledge downside risk.

Layer map (PLAN.md sec.7.1). Each layer owns one job and never does the next
layer's:

    ingest/      network I/O, retries, raw persistence   (never interprets)
    data/        the point-in-time join rule             (never computes features)
    features/    derived columns                          (never touches raw files)
    labels/      forward drawdown                         (never fits)
    training/    fold splitting, fitting, selection       (never persists predictions)
    evaluation/  metrics, backtest, leakage proofs        (never fits)
    explain/     SHAP global + local + text
    inference/   THE single scoring path                  (never trains)
    api/         transport only                           (no business logic)
    db/          all SQL, one module
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
