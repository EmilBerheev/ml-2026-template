"""Живая проверка сервиса: запускается грейдером в отдельном процессе.

Строит запрос к /predict по OpenAPI-схеме вашего приложения и первой подходящей строке
data/sample.csv, печатает JSON с результатами. Ваш код здесь не меняется.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys


def field_type(spec: dict, components: dict) -> tuple[str, list | None]:
    if "$ref" in spec:
        spec = components[spec["$ref"].split("/")[-1]]
    if "anyOf" in spec:
        for alt in spec["anyOf"]:
            if alt.get("type") != "null":
                return field_type(alt, components)
    return spec.get("type", "string"), spec.get("enum")


def cast(value: str, kind: str):
    if value == "" or value is None:
        return None
    if kind == "integer":
        f = float(value)
        return int(f) if not math.isnan(f) else None
    if kind == "number":
        f = float(value)
        return None if math.isnan(f) else f
    if kind == "boolean":
        return value.strip().lower() in ("1", "true", "yes")
    return value


def main() -> None:
    from fastapi.testclient import TestClient

    from mlproject.api import app

    sample_path = sys.argv[1]
    result: dict = {}
    with TestClient(app) as client:
        r = client.get("/health")
        result["health"] = {"status": r.status_code, "body": _body(r)}
        schema = app.openapi()
        op = schema["paths"].get("/predict", {}).get("post")
        if not op:
            result["error"] = "нет POST /predict"
            print(json.dumps(result, ensure_ascii=False, default=str))
            return
        components = schema.get("components", {}).get("schemas", {})
        ref = op["requestBody"]["content"]["application/json"]["schema"]
        model = components[ref["$ref"].split("/")[-1]] if "$ref" in ref else ref
        props, required = model.get("properties", {}), set(model.get("required", []))
        types = {k: field_type(v, components) for k, v in props.items()}
        result["constraints"] = sum(
            1 for v in props.values()
            if any(k in json.dumps(v) for k in ('"minimum"', '"maximum"', '"enum"', '"const"',
                                                 '"maxLength"', '"exclusiveMinimum"')))
        payload = None
        with open(sample_path, encoding="utf-8", errors="replace") as fh:
            for row in csv.DictReader(fh):
                cand = {}
                for k in props:
                    if k in row:
                        v = cast(row[k], types[k][0])
                        if v is not None:
                            cand[k] = v
                if required <= set(cand):
                    payload = cand
                    break
        if payload is None:
            result["error"] = ("не нашёл в sample.csv строку со всеми обязательными полями "
                               f"{sorted(required)}")
        else:
            r = client.post("/predict", json=payload)
            result["predict"] = {"status": r.status_code, "body": _body(r), "payload": payload}
            bad = dict(payload)
            numeric = [k for k, (t, _) in types.items() if t in ("integer", "number")]
            if numeric:
                bad[numeric[0]] = "не число"
            else:
                bad = {}
            r = client.post("/predict", json=bad)
            result["invalid"] = {"status": r.status_code}
    print(json.dumps(result, ensure_ascii=False, default=str))


def _body(r):
    try:
        return r.json()
    except ValueError:
        return r.text[:200]


if __name__ == "__main__":
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    main()
