# Copyright (c) 2026 James H. Smith. MIT License.
"""Flask app factory and entry point."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from flask import Flask


def create_app() -> Flask:
    """Create and configure the Flask application."""
    load_dotenv()

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    app = Flask(
        __name__,
        template_folder=os.path.join(root_dir, "templates"),
        static_folder=os.path.join(root_dir, "static"),
    )
    app.config["SECRET_KEY"] = os.urandom(24)
    app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024  # 1 MB

    from telephone.routes import bp

    app.register_blueprint(bp)

    @app.after_request
    def set_security_headers(response):
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self'"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        return response

    return app


def main() -> None:
    app = create_app()
    app.run(host="0.0.0.0", port=5000)


if __name__ == "__main__":
    main()
