"""Integration-test configuration.

No event-loop override: the gateway (incl. the decision log) must run on the default
loop of every platform, i.e. the ProactorEventLoop on Windows.
"""
