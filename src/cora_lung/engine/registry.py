"""Experiment registry for CORA-Lung."""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json


def utc_now():
    return datetime.now(
        timezone.utc
    ).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


class ExperimentRegistry:

    def __init__(
        self,
        path,
    ):
        self.path = Path(
            path
        )

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _read(
        self,
    ):
        if not self.path.exists():
            return {
                "runs": []
            }

        return json.loads(
            self.path.read_text(
                encoding="utf-8"
            )
        )

    def _write(
        self,
        payload,
    ):
        tmp = self.path.with_suffix(
            self.path.suffix
            + ".tmp"
        )

        tmp.write_text(
            json.dumps(
                payload,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        tmp.replace(
            self.path
        )

    def register(
        self,
        run_id,
        metadata,
    ):
        payload = self._read()

        existing = {
            run[
                "run_id"
            ]
            for run in payload[
                "runs"
            ]
        }

        if run_id in existing:
            raise RuntimeError(
                f"Run already registered: {run_id}"
            )

        payload[
            "runs"
        ].append(
            {
                "run_id":
                    run_id,

                "status":
                    "CREATED",

                "created_at_utc":
                    utc_now(),

                "updated_at_utc":
                    utc_now(),

                **metadata,
            }
        )

        self._write(
            payload
        )

    def update(
        self,
        run_id,
        **changes,
    ):
        payload = self._read()

        found = False

        for run in payload[
            "runs"
        ]:
            if run[
                "run_id"
            ] == run_id:
                run.update(
                    changes
                )

                run[
                    "updated_at_utc"
                ] = utc_now()

                found = True

                break

        if not found:
            raise RuntimeError(
                f"Unknown run: {run_id}"
            )

        self._write(
            payload
        )
