"""
generate_sample_data.py

Produces a SCHEMA-ACCURATE SAMPLE of the PaySim mobile-money fraud dataset
so the rest of the pipeline can be built and tested immediately, without
needing network access to download the real ~6.3M-row dataset.

*** THIS IS NOT THE REAL DATASET. ***
Replace data/raw/paysim_transactions.csv with the real file before
publishing any results or putting numbers on your resume.

Real dataset (PaySim, ~6.3M rows, synthetic but widely used as a fraud-
detection benchmark, based on real mobile money transaction logs from an
African mobile money provider):
https://www.kaggle.com/datasets/ealaxi/paysim1

Why PaySim over IEEE-CIS for this project: PaySim's schema (transaction
type, sender/receiver balances, mobile-money transfers) is thematically
closer to a payments-infrastructure company like NPCI than IEEE-CIS's
anonymized credit-card features -- easier to speak to in an interview
about UPI-style payment rails specifically.

Citation:
Lopez-Rojas, E., Elmir, A., Axelsson, S. (2016). PaySim: A financial
mobile money simulator for fraud detection. 28th European Modeling and
Simulation Symposium (EMSS).
"""

import random
import csv
from pathlib import Path

random.seed(42)

TXN_TYPES = ["CASH_OUT", "PAYMENT", "CASH_IN", "TRANSFER", "DEBIT"]
# Real PaySim fraud is concentrated in TRANSFER and CASH_OUT only
FRAUD_ELIGIBLE_TYPES = {"TRANSFER", "CASH_OUT"}

HEADER = ["step", "type", "amount", "nameOrig", "oldbalanceOrg",
          "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest",
          "isFraud", "isFlaggedFraud"]


def make_row(step):
    txn_type = random.choices(
        TXN_TYPES, weights=[22, 34, 22, 10, 12]
    )[0]

    amount = round(random.lognormvariate(6, 2), 2)  # skewed, realistic-ish
    old_orig = round(random.uniform(0, 50000), 2)
    new_orig = max(0, round(old_orig - amount, 2)) if txn_type in ("CASH_OUT", "PAYMENT", "TRANSFER", "DEBIT") else old_orig
    old_dest = round(random.uniform(0, 50000), 2)
    new_dest = round(old_dest + amount, 2) if txn_type in ("CASH_OUT", "TRANSFER", "CASH_IN") else old_dest

    name_orig = f"C{random.randint(10**8, 10**9 - 1)}"
    name_dest = (f"M{random.randint(10**8, 10**9 - 1)}" if txn_type == "PAYMENT"
                 else f"C{random.randint(10**8, 10**9 - 1)}")

    # Fraud only occurs in TRANSFER/CASH_OUT, and is rare (~0.13% in real PaySim)
    is_fraud = 0
    if txn_type in FRAUD_ELIGIBLE_TYPES and random.random() < 0.0013:
        is_fraud = 1
        # Real PaySim fraud pattern: empties the origin account
        new_orig = 0.0

    is_flagged = 1 if (txn_type == "TRANSFER" and amount > 200000) else 0

    return [step, txn_type, amount, name_orig, old_orig, new_orig,
            name_dest, old_dest, new_dest, is_fraud, is_flagged]


if __name__ == "__main__":
    out_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "paysim_transactions.csv"
    n_rows = 50_000
    n_steps = 30  # simulate 30 "hours" of activity, PaySim uses ~744 steps (1 month hourly)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        for _ in range(n_rows):
            step = random.randint(1, n_steps)
            writer.writerow(make_row(step))

    print(f"Wrote {n_rows} SAMPLE rows to {out_path}")
    print("Replace this file with the real PaySim dataset (~6.3M rows) before using results externally.")
    print("Download: https://www.kaggle.com/datasets/ealaxi/paysim1")
