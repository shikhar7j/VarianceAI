from __future__ import annotations

import logging
import os

from flask import Flask


def create_app() -> Flask:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    static_dir = os.path.join(base_dir, "static")

    app = Flask(__name__, static_folder=static_dir, static_url_path="")

    from app.routes import bp

    app.register_blueprint(bp)

    return app