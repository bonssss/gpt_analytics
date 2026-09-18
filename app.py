import os
import sys

# Check if running inside Streamlit script runner
try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    inside_streamlit = get_script_run_ctx() is not None
except Exception:
    inside_streamlit = False

if inside_streamlit:
    # Executed via `streamlit run app.py`
    import app.dashboard
else:
    # Executed via `python app.py` (e.g. Alet Cloud / PaaS container entrypoint)
    import streamlit.web.cli as stcli

    port = os.environ.get("PORT", "3000")
    script_path = os.path.join(os.path.dirname(__file__), "app", "dashboard.py")
    
    sys.argv = [
        "streamlit",
        "run",
        script_path,
        "--server.port",
        str(port),
        "--server.address",
        "0.0.0.0",
        "--server.headless",
        "true"
    ]
    sys.exit(stcli.main())
