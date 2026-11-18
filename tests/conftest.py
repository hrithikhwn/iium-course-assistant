"""Unit tests run with the default settings, whatever .env says.

app/config.py reads .env when it is imported, so a QUERY_REWRITE=false or
SEARCH_GUARANTEED=0 there would change what the tests check. Variables that
are already set win over .env, so setting them here first pins them.
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
