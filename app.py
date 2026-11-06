"""WSGI entrypoint for hosting the CellSense dashboard (e.g. Vercel's Python runtime).

Vercel imports this module and serves the WSGI application named ``app``. Serverless functions are
frozen between requests, so the simulation advances on each request instead of in a background thread.

Local development uses the CLI instead:  python -m cellsense dashboard
"""
from cellsense.dashboard.app import create_app
from cellsense.dashboard.session import LiveSession

session = LiveSession(mode="demo", speed=5.0, warmup_s=600, background=False)
dash_app = create_app(session)
app = dash_app.server  # the Flask WSGI app Dash runs on

if __name__ == "__main__":
    dash_app.run(host="127.0.0.1", port=8050, debug=False)
