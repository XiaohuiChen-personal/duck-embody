"""T0.1 smoke test: the vendored policy artifacts are the ones the docs describe.

Runs offline (no kit app). Asserts, against ``policy/``:

* ``model_2999.pt`` loads on CPU and its actor has the 59-dim input contract
  documented in doc 02 §2 (obs normalizer mean shape (1, 59); first MLP layer
  512×59).
* ``params/env.yaml`` parses and its *policy* observation group contains no
  ``base_lin_vel`` term — that dim is critic-only (62-dim critic), and its
  presence would mean we vendored the wrong run.

  ``yaml.unsafe_load`` is required, not sloppiness: Isaac Lab dumps configs with
  ``!!python/tuple`` / ``!!python/object`` tags and ``safe_load`` raises
  ``ConstructorError`` on them (verified 2026-07-26). The file is a repo-local
  artifact we vendored ourselves, so the trust boundary is fine.
* ``params/agent.yaml`` declares ``obs_normalization: true`` (doc 02 §2's
  "never pre-normalize" warning depends on it) and the asymmetric obs_groups.
* every file matches ``policy/checksums.txt``.

Run:  ~/IsaacLab/isaaclab.sh -p scripts/smoke_policy_artifacts.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch
import yaml

POLICY_DIR = Path(__file__).resolve().parent.parent / "policy"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    failures: list[str] = []

    def check(label: str, condition: bool, detail: str = "") -> None:
        print(f"  {'PASS' if condition else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""))
        if not condition:
            failures.append(label)

    print("== checksums ==")
    for line in (POLICY_DIR / "checksums.txt").read_text().splitlines():
        expected, name = line.split()
        actual = _sha256(POLICY_DIR / name)
        check(f"sha256 {name}", actual == expected, actual[:16])

    print("\n== checkpoint (model_2999.pt) ==")
    ckpt = torch.load(POLICY_DIR / "model_2999.pt", map_location="cpu", weights_only=False)
    print(f"  top-level keys: {list(ckpt.keys())}")
    # rsl-rl-lib 5.x stores the actor under its own key (NOT "model_state_dict").
    actor_sd = ckpt["actor_state_dict"]

    check("training iteration == 2999", ckpt.get("iter") == 2999, str(ckpt.get("iter")))

    mean_shape = tuple(actor_sd["obs_normalizer._mean"].shape)
    check("obs_normalizer._mean shape == (1, 59)", mean_shape == (1, 59), str(mean_shape))

    w0_shape = tuple(actor_sd["mlp.0.weight"].shape)
    check("actor mlp.0.weight == (512, 59)", w0_shape == (512, 59), str(w0_shape))

    # doc 02 §1: hidden layers 512/256/128, 16 joint outputs.
    wout_shape = tuple(actor_sd["mlp.6.weight"].shape)
    check("actor mlp.6.weight == (16, 128)", wout_shape == (16, 128), str(wout_shape))

    print("\n== params/env.yaml (obs contract) ==")
    env_cfg = yaml.unsafe_load((POLICY_DIR / "params" / "env.yaml").read_text())
    policy_group = env_cfg["observations"]["policy"]
    terms = [k for k, v in policy_group.items() if isinstance(v, dict) and "func" in v]
    print(f"  policy obs terms: {terms}")
    check(
        "policy group has NO base_lin_vel",
        policy_group.get("base_lin_vel") is None,
        repr(policy_group.get("base_lin_vel")),
    )
    critic_group = env_cfg["observations"].get("critic") or {}
    check(
        "critic group HAS base_lin_vel (asymmetric actor/critic)",
        critic_group.get("base_lin_vel") is not None,
    )
    check("dt == 0.005", env_cfg["sim"]["dt"] == 0.005, str(env_cfg["sim"]["dt"]))
    check("decimation == 4", env_cfg["decimation"] == 4, str(env_cfg["decimation"]))

    ranges = env_cfg["commands"]["base_velocity"]["ranges"]
    print(f"  training command hull: {ranges}")

    print("\n== params/agent.yaml ==")
    agent_cfg = yaml.unsafe_load((POLICY_DIR / "params" / "agent.yaml").read_text())
    # rsl-rl-lib 5.x schema: separate top-level `actor:` / `critic:` blocks and a
    # top-level `obs_groups:`; the legacy `policy:` key is present but empty.
    # This is exactly the rename that makes `handle_deprecated_rsl_rl_cfg`
    # mandatory when constructing OnPolicyRunner (doc 02 §3).
    check(
        "actor.obs_normalization is true (never pre-normalize)",
        agent_cfg["actor"]["obs_normalization"] is True,
    )
    check(
        "critic.obs_normalization is true",
        agent_cfg["critic"]["obs_normalization"] is True,
    )
    check(
        "obs_groups is asymmetric {actor:[policy], critic:[critic]}",
        agent_cfg["obs_groups"] == {"actor": ["policy"], "critic": ["critic"]},
        str(agent_cfg["obs_groups"]),
    )
    check("runner class_name == OnPolicyRunner", agent_cfg["class_name"] == "OnPolicyRunner")
    print(f"  actor hidden_dims: {agent_cfg['actor']['hidden_dims']}")

    print("\n== result ==")
    if failures:
        for f in failures:
            print(f"  FAILED: {f}")
        return 1
    print("  OK - vendored artifacts match the doc 02 contract")
    return 0


if __name__ == "__main__":
    sys.exit(main())
