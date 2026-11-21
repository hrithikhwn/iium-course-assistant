"""Unit tests run with these settings, whatever .env says.

app/config.py reads .env when it is imported, so a GRADING=false or
SEARCH_GUARANTEED=0 there would change what the tests check. Variables that
are already set win over .env, so setting them here first pins them.

They are config.py's defaults, except QUERY_REWRITE: it is off by default
since the Phase 4 eval, but on here so the agent tests cover the plan step
too (tests of the default pass rewrite=False). test_config.py checks the
defaults themselves.
"""
import os

os.environ.update({
    "QUERY_REWRITE": "true",
    "GRADING": "true",
    "REFUSE_ON_FAILED_GRADE": "false",
    "TOP_K": "5",
    "MAX_RETRIES": "2",
    "SEARCH_GUARANTEED": "2",
    "MAX_CONTEXT_CHARS": "11000",
})
