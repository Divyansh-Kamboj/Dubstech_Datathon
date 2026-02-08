"""
Safety Net Simulator 2026 — Budget Allocation Engine
=====================================================
Ranks patient groups by a blended priority score (clinical risk vs.
cost-efficiency) and greedily funds groups until the budget is exhausted.
"""


class BudgetAllocator:
    """Decides which patient groups receive funding under a fixed budget."""

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------
    def run_allocation(self, df, total_budget, humanity_weight=0.6):
        """
        Allocate a fixed budget across patient groups.

        Parameters
        ----------
        df : pandas.DataFrame
            Must contain ``Risk_Score``, ``Cost_Per_Person``,
            ``Total_Group_Cost``, and ``Projected_Volume``.
        total_budget : float
            Total dollars available for allocation.
        humanity_weight : float, default 0.6
            Weight given to clinical risk (0-1).  The remainder
            ``(1 - humanity_weight)`` is given to cost-efficiency.

        Returns
        -------
        (pandas.DataFrame, float)
            The enriched DataFrame (with ``Normalized_Risk``,
            ``Normalized_Efficiency``, ``Priority_Score``, and
            ``Funded_Status`` columns) and the unspent budget.
        """
        df = df.copy()

        # 1. Normalize inputs ------------------------------------------------
        df["Normalized_Risk"] = self._min_max_normalize(df["Risk_Score"])

        # For efficiency, *invert* so that the cheapest → 1, priciest → 0
        df["Normalized_Efficiency"] = self._min_max_normalize(
            df["Cost_Per_Person"], invert=True
        )

        # 2. Blended priority score ------------------------------------------
        df["Priority_Score"] = (
            humanity_weight * df["Normalized_Risk"]
            + (1 - humanity_weight) * df["Normalized_Efficiency"]
        ).round(4)

        # 3. Greedy allocation ------------------------------------------------
        df = df.sort_values("Priority_Score", ascending=False).reset_index(
            drop=True
        )

        remaining_budget = float(total_budget)
        funded = []

        for cost in df["Total_Group_Cost"]:
            if cost <= remaining_budget:
                funded.append("Yes")
                remaining_budget -= cost
            else:
                funded.append("No")

        df["Funded_Status"] = funded

        return df, round(remaining_budget, 2)

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------
    @staticmethod
    def _min_max_normalize(series, invert=False):
        """Scale *series* to [0, 1].  If *invert*, flip so max → 0."""
        s_min = series.min()
        s_max = series.max()
        if s_max == s_min:
            return series.map(lambda _: 0.5)
        normalized = (series - s_min) / (s_max - s_min)
        return (1 - normalized).round(4) if invert else normalized.round(4)
