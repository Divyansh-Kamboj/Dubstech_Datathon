"""
Project Aesclepius — 2026 Simulator Dashboard
==============================================
Streamlit front-end for the Safety Net budget-allocation engine.

Design note: there is NO button gate.  The solver runs reactively
every time a sidebar slider or input changes, which is the standard
Streamlit pattern for dashboards.
"""

import streamlit as st
import pandas as pd
import altair as alt
