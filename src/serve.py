
"""HTTP server for the containerized Iris classifier."""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = "models/final_model.joblib"
ENCODER_PATH = "models/label_encoder.joblib"

FEATURE_ORDER = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
    "sepal_area",
    "petal_area",
    "sepal_to_petal_length_ratio",
]

model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)


class PredictionHandler(BaseHTTPRequestHandler):
    """Handle health checks and Iris prediction requests."""

    def _respond(self, status, payload):
        """Send a JSON response with the specified HTTP status."""
        response = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def do_GET(self):
        """Handle health-check requests."""
        if self.path == "/health":
            self._respond(200, {"status": "healthy"})
        else:
            self._respond(404, {"error": "not found"})

    def do_POST(self):
        """Validate input and return an Iris prediction."""
        if self.path != "/predict":
            self._respond(404, {"error": "not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                self._respond(400, {"error": "Request body is empty"})
                return

            body = json.loads(self.rfile.read(length))

            if not isinstance(body, dict):
                self._respond(
                    400,
                    {"error": "JSON body must be an object"},
                )
                return

            missing = [
                name for name in FEATURE_ORDER if name not in body
            ]
            if missing:
                self._respond(
                    400,
                    {
                        "error": "Missing features",
                        "missing": missing,
                    },
                )
                return

            values = [float(body[name]) for name in FEATURE_ORDER]

            if not np.isfinite(values).all():
                self._respond(
                    400,
                    {"error": "Feature values must be finite numbers"},
                )
                return

            features = pd.DataFrame(
                [values],
                columns=FEATURE_ORDER,
            )

        except (
            json.JSONDecodeError,
            ValueError,
            TypeError,
            OverflowError,
        ):
            self._respond(
                400,
                {"error": "Invalid JSON or feature values"},
            )
            return

        prediction_index = model.predict(features)[0]
        prediction = label_encoder.inverse_transform(
            [prediction_index]
        )[0]
        probabilities = model.predict_proba(features)[0].tolist()

        self._respond(
            200,
            {
                "prediction": prediction,
                "probabilities": probabilities,
            },
        )


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", 8000), PredictionHandler)
    print("Iris server listening on port 8000", flush=True)
    server.serve_forever()
