# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from flask import Flask
from importlib import import_module
from flask import render_template, request
from jinja2 import TemplateNotFound


def create_app(config):
    app = Flask(__name__)
    app.config.from_object(config)

    @app.route("/index")
    def index():

        return render_template("home/index.html", segment="index")

    @app.route("/<template>")
    def route_template(template):

        try:

            if not template.endswith(".html"):
                template += ".html"

            # Detect the current page
            segment = get_segment(request)

            # Serve the file (if exists) from app/templates/home/FILE.html
            return render_template("home/" + template, segment=segment)

        except TemplateNotFound:
            return render_template("home/page-404.html"), 404

        except:
            return render_template("home/page-500.html"), 500

    return app


# Helper - Extract current page name from request
def get_segment(request):
    try:
        segment = request.path.split("/")[-1]
        if segment == "":
            segment = "index"
        return segment
    except:
        return None
