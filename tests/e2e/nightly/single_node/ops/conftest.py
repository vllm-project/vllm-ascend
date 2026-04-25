import time
from datetime import datetime
import pytest


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Hook to add timestamp to test reports"""
    start_time = datetime.now().strftime("[%H:%M:%S]")
    
    outcome = yield
    
    report = outcome.get_result()
    
    if report.when == 'call':
        
        print(f"{start_time}")