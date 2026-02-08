"""
Safety Net Simulator 2026 — Budget Allocation Engine
=====================================================
Ranks patient groups by a blended priority score (clinical risk vs.
cost-efficiency) and greedily funds groups — with **fractional funding**
so that even groups larger than the remaining budget receive a partial
allocation.

Normalization
-------------
- **Risk**       : ``Risk_Score / 10``  →  0.0 – 1.0 (higher = sicker)
- **Efficiency** : ``(max_cost - cost) / (max_cost - min_cost)``
                   →  0.0 (most expensive) – 1.0 (cheapest)
- **Priority**   : ``w_efficiency × Efficiency + w_humanity × Risk``
"""

import numpy as np


class BudgetAllocator:
    """Decides which patient groups receive funding under a fixed budget."""

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------
    def run_allocation(
        self,
        df,
        total_budget,
        w_efficiency=0.5,
        w_humanity=0.5,
    ):
        """
        Allocate a fixed budget across patient groups.

        Parameters
        ----------
        df : pandas.DataFrame
            Must contain ``Risk_Score``, ``Cost_Per_Person``,
            ``Total_Group_Cost``, and ``Projected_Volume``.
        total_budget : float
            Total dollars available for allocation.
        w_efficiency : float
            Weight given to cost-efficiency (0-1).
        w_humanity : float
            Weight given to clinical risk (0-1).

        Returns
        -------
        (pandas.DataFrame, float)
            The enriched DataFrame (with scoring, funding, and
            coverage columns) and the unspent budget.
        """
        df = df.copy()

        # 1. Normalize Risk — divide by 10 so it sits in [0, 1] -----------
        df["Normalized_Risk"] = (df["Risk_Score"] / 10.0).round(4)

        # 2. Normalize Efficiency — inverted min-max on cost ---------------
        max_cost = df["Cost_Per_Person"].max()
        min_cost = df["Cost_Per_Person"].min()
        if max_cost == min_cost:
            df["Normalized_Efficiency"] = 0.5
        else:
            df["Normalized_Efficiency"] = (
                (max_cost - df["Cost_Per_Person"]) / (max_cost - min_cost)
            ).round(4)

        # 3. Blended priority score ----------------------------------------
        df["Priority_Score"] = (
            w_efficiency * df["Normalized_Efficiency"]
            + w_humanity * df["Normalized_Risk"]
        ).round(4)

        # 4. Greedy allocation with fractional funding ---------------------
        df = df.sort_values("Priority_Score", ascending=False).reset_index(
            drop=True
        )

        remaining_budget = float(total_budget)
        funded_pct = []
        people_covered = []
        amount_allocated = []

        for _, row in df.iterrows():
            cost = row["Total_Group_Cost"]
            volume = row["Projected_Volume"]

            if remaining_budget <= 0:
                funded_pct.append(0.0)
                people_covered.append(0)
                amount_allocated.append(0.0)
            elif cost <= remaining_budget:
                funded_pct.append(1.0)
                people_covered.append(int(volume))
                amount_allocated.append(round(cost, 2))
                remaining_budget -= cost
            else:
                frac = remaining_budget / cost if cost > 0 else 0.0
                funded_pct.append(round(frac, 6))
                people_covered.append(int(np.floor(volume * frac)))
                amount_allocated.append(round(remaining_budget, 2))
                remaining_budget = 0.0

        df["Funded_Pct"] = funded_pct
        df["People_Covered"] = people_covered
        df["Amount_Allocated"] = amount_allocated

        df["Funded_Status"] = df["Funded_Pct"].apply(
            lambda p: "Yes" if p >= 1.0 else ("Partial" if p > 0 else "No")
        )

        return df, round(remaining_budget, 2)
