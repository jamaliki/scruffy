"""Resubmit the six jobs lost during the Scruffy allocation handover.

This reuses each already-materialized Koochak launch manifest.  The Kaveh
training entrypoints discover the newest ready checkpoint in their existing
output directories; the Modelangelo manifest retains its explicit step-
107000 resume path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from koochak.jobs import PreparedRun, submit_scruffy
from scruffy import ResourceRequest


ROOT = Path("/mnt/gbi-shared/home/kiarash-jamali/.scruffy/queues/263105")
ALLOCATION = "289717"


@dataclass(frozen=True)
class RecoveryJob:
    project_id: str
    request_id: str
    manifest_path: Path
    resources: ResourceRequest
    workflow_id: str | None = None
    task_id: str | None = None
    needs: tuple[dict[str, str], ...] = ()
    checkpoint_path: Path | None = None
    ready_manifest_path: Path | None = None
    explicit_resume_path: Path | None = None


JOBS = (
    RecoveryJob(
        project_id="kaveh-ce20-20260806",
        request_id=(
            "hk-inverse-folding-b256-factorial-3x400k-clean-e1af3e5-v1/"
            "train-q025_cb_reset/recovery-v2"
        ),
        manifest_path=Path(
            "/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-runs/"
            "inverse-folding-b256-factorial-3x400k-clean/"
            "e1af3e5f93c047900ce13420b48571429963b23a/v1/train/"
            "q025_cb_reset/launch.json"
        ),
        resources=ResourceRequest(1, 1, 14, 240, 200000),
        workflow_id="hk-inverse-folding-b256-factorial-3x400k-clean-e1af3e5-v1",
        task_id="train-q025_cb_reset",
        needs=(
            {"task_id": "canary-training", "condition": "succeeded"},
            {"task_id": "canary-packing", "condition": "succeeded"},
        ),
        checkpoint_path=Path(
            "/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-runs/"
            "inverse-folding-b256-factorial-3x400k-clean/"
            "e1af3e5f93c047900ce13420b48571429963b23a/v1/train/"
            "q025_cb_reset/step000130000.pt"
        ),
        ready_manifest_path=Path(
            "/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-runs/"
            "inverse-folding-b256-factorial-3x400k-clean/"
            "e1af3e5f93c047900ce13420b48571429963b23a/v1/train/"
            "q025_cb_reset/step000130000.pt.ready.json"
        ),
    ),
    RecoveryJob(
        project_id="kaveh-ce20-20260806",
        request_id=(
            "hk-inverse-folding-b256-factorial-3x400k-clean-e1af3e5-v1/"
            "train-q025_cb_fixed/recovery-v2"
        ),
        manifest_path=Path(
            "/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-runs/"
            "inverse-folding-b256-factorial-3x400k-clean/"
            "e1af3e5f93c047900ce13420b48571429963b23a/v1/train/"
            "q025_cb_fixed/launch.json"
        ),
        resources=ResourceRequest(1, 1, 14, 240, 200000),
        workflow_id="hk-inverse-folding-b256-factorial-3x400k-clean-e1af3e5-v1",
        task_id="train-q025_cb_fixed",
        needs=(
            {"task_id": "canary-training", "condition": "succeeded"},
            {"task_id": "canary-packing", "condition": "succeeded"},
        ),
        checkpoint_path=Path(
            "/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-runs/"
            "inverse-folding-b256-factorial-3x400k-clean/"
            "e1af3e5f93c047900ce13420b48571429963b23a/v1/train/"
            "q025_cb_fixed/step000130000.pt"
        ),
        ready_manifest_path=Path(
            "/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-runs/"
            "inverse-folding-b256-factorial-3x400k-clean/"
            "e1af3e5f93c047900ce13420b48571429963b23a/v1/train/"
            "q025_cb_fixed/step000130000.pt.ready.json"
        ),
    ),
    RecoveryJob(
        project_id="kaveh-ce20-20260806",
        request_id=(
            "hk-inverse-folding-b256-factorial-3x400k-clean-e1af3e5-v1/"
            "train-q050_cb_reset/recovery-v2"
        ),
        manifest_path=Path(
            "/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-runs/"
            "inverse-folding-b256-factorial-3x400k-clean/"
            "e1af3e5f93c047900ce13420b48571429963b23a/v1/train/"
            "q050_cb_reset/launch.json"
        ),
        resources=ResourceRequest(1, 1, 14, 240, 200000),
        workflow_id="hk-inverse-folding-b256-factorial-3x400k-clean-e1af3e5-v1",
        task_id="train-q050_cb_reset",
        needs=(
            {"task_id": "canary-training", "condition": "succeeded"},
            {"task_id": "canary-packing", "condition": "succeeded"},
        ),
        checkpoint_path=Path(
            "/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-runs/"
            "inverse-folding-b256-factorial-3x400k-clean/"
            "e1af3e5f93c047900ce13420b48571429963b23a/v1/train/"
            "q050_cb_reset/step000130000.pt"
        ),
        ready_manifest_path=Path(
            "/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-runs/"
            "inverse-folding-b256-factorial-3x400k-clean/"
            "e1af3e5f93c047900ce13420b48571429963b23a/v1/train/"
            "q050_cb_reset/step000130000.pt.ready.json"
        ),
    ),
    RecoveryJob(
        project_id="modelangelo",
        request_id=(
            "modelangelo-gnn-sequence-residual-dropout-scratch-4gpu-"
            "resume-step107000-20260825-recovery-v2"
        ),
        manifest_path=Path(
            "/mnt/lustre/users/kiarash-eitgbi/runs/"
            "gnn_sequence_residual_dropout_scratch_4gpu_modelangelo_"
            "resume_step107000_20260825_v1/launch.json"
        ),
        resources=ResourceRequest(1, 4, 56, 512),
        explicit_resume_path=Path(
            "/mnt/lustre/users/kiarash-eitgbi/runs/"
            "gnn_sequence_residual_dropout_scratch_4gpu_modelangelo_"
            "resume_step020000_20260821_v1/gnn-sequence-residual-dropout-"
            "scratch-4gpu/step000107000.pt"
        ),
    ),
    RecoveryJob(
        project_id="kaveh-ce20-20260806",
        request_id=(
            "hk-signal-pair-recycling-decay08-b256-400k-88331b6-v1/"
            "train-initial/recovery-v2"
        ),
        manifest_path=Path(
            "/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-runs/"
            "signal-pair-recycling-decay08-b256-400k/"
            "88331b6ab9a37f19fed3c6338583c06f456743c3/v1/jobs/"
            "train-initial/launch.json"
        ),
        resources=ResourceRequest(1, 1, 14, 240, 118000),
        workflow_id="hk-signal-pair-recycling-decay08-b256-400k-88331b6-v1",
        task_id="train-initial",
        needs=({"task_id": "canary-compile", "condition": "succeeded"},),
        checkpoint_path=Path(
            "/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-runs/"
            "signal-pair-recycling-decay08-b256-400k/"
            "88331b6ab9a37f19fed3c6338583c06f456743c3/v1/train/"
            "decay08_register_pair/step000270000.pt"
        ),
        ready_manifest_path=Path(
            "/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-runs/"
            "signal-pair-recycling-decay08-b256-400k/"
            "88331b6ab9a37f19fed3c6338583c06f456743c3/v1/train/"
            "decay08_register_pair/step000270000.pt.ready.json"
        ),
    ),
    RecoveryJob(
        project_id="kaveh-ce20-20260806",
        request_id=(
            "hk-signal-propagation-fixed-b256-400k-720be68-v1/"
            "train/recovery-v2"
        ),
        manifest_path=Path(
            "/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-runs/"
            "signal-propagation-fixed-b256-400k/"
            "720be6864166908d4bda57cd6e59b0954090d5ef/v1/train/"
            "batch256_signal_fixed/launch.json"
        ),
        resources=ResourceRequest(1, 1, 14, 240, 138000),
        workflow_id="hk-signal-propagation-fixed-b256-400k-720be68-v1",
        task_id="train",
        checkpoint_path=Path(
            "/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-runs/"
            "signal-propagation-fixed-b256-400k/"
            "720be6864166908d4bda57cd6e59b0954090d5ef/v1/train/"
            "batch256_signal_fixed/step000310000.pt"
        ),
        ready_manifest_path=Path(
            "/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-runs/"
            "signal-propagation-fixed-b256-400k/"
            "720be6864166908d4bda57cd6e59b0954090d5ef/v1/train/"
            "batch256_signal_fixed/step000310000.pt.ready.json"
        ),
    ),
)


def prepared_from_manifest(manifest_path: Path) -> PreparedRun:
    manifest_content = manifest_path.read_bytes()
    document = json.loads(manifest_content)
    environment = document["environment"]
    return PreparedRun(
        name=document["name"],
        cwd=document["cwd"],
        run_dir=document["run_dir"],
        python=environment["python"],
        manifest_path=str(manifest_path),
        manifest_sha256=hashlib.sha256(manifest_content).hexdigest(),
        artifacts=(),
    )


def validate(job: RecoveryJob) -> PreparedRun:
    if not job.manifest_path.is_file():
        raise FileNotFoundError(job.manifest_path)
    if job.checkpoint_path is not None:
        if not job.checkpoint_path.is_file():
            raise FileNotFoundError(job.checkpoint_path)
        if job.ready_manifest_path is None or not job.ready_manifest_path.is_file():
            raise FileNotFoundError(job.ready_manifest_path or "ready manifest")
    if job.explicit_resume_path is not None and not job.explicit_resume_path.is_file():
        raise FileNotFoundError(job.explicit_resume_path)
    return prepared_from_manifest(job.manifest_path)


def submit(job: RecoveryJob, prepared: PreparedRun) -> dict[str, object]:
    return submit_scruffy(
        prepared,
        root=ROOT,
        resources=job.resources,
        request_id=job.request_id,
        project_id=job.project_id,
        workflow_id=job.workflow_id,
        task_id=job.task_id,
        needs=list(job.needs),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for job in JOBS:
        prepared = validate(job)
        print(
            f"{job.project_id} {prepared.name} {job.request_id} "
            f"resources={job.resources.to_dict()}"
        )
        if not args.dry_run:
            result = submit(job, prepared)
            print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
