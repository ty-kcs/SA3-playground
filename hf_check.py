"""HuggingFace token validation and model-access checks for the playground."""

from __future__ import annotations

from dataclasses import dataclass

from huggingface_hub import HfApi, login
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

# Models referenced in sa3_playground.ipynb (+ SAME-L for medium).
PLAYGROUND_REPOS: tuple[tuple[str, str], ...] = (
    ("stabilityai/stable-audio-3-medium", "medium"),
    ("stabilityai/stable-audio-3-medium-base", "medium-base"),
    ("stabilityai/SAME-L", "SAME-L autoencoder (medium)"),
)


@dataclass(frozen=True)
class RepoCheck:
    repo_id: str
    label: str
    ok: bool
    message: str


def verify_hf_token(token: str) -> dict:
    token = (token or "").strip()
    if not token:
        raise ValueError("Token is empty.")
    return HfApi().whoami(token=token)


def check_repo_access(token: str, repo_id: str, label: str) -> RepoCheck:
    api = HfApi()
    page = f"https://huggingface.co/{repo_id}"
    try:
        api.model_info(repo_id, token=token)
        return RepoCheck(repo_id, label, True, "access OK")
    except GatedRepoError:
        return RepoCheck(
            repo_id,
            label,
            False,
            f"gated — accept the license on the model page: {page}",
        )
    except RepositoryNotFoundError:
        return RepoCheck(repo_id, label, False, f"repo not found: {page}")
    except HfHubHTTPError as exc:
        code = exc.response.status_code if exc.response is not None else None
        if code == 401:
            return RepoCheck(repo_id, label, False, "unauthorized — invalid or expired token")
        if code == 403:
            return RepoCheck(
                repo_id,
                label,
                False,
                f"access denied — accept the license: {page}",
            )
        return RepoCheck(repo_id, label, False, f"HTTP {code}: {exc}")
    except Exception as exc:  # noqa: BLE001 — surface unexpected HF errors in UI
        return RepoCheck(repo_id, label, False, str(exc))


def check_playground_access(token: str) -> tuple[dict, list[RepoCheck]]:
    whoami = verify_hf_token(token)
    checks = [check_repo_access(token, repo_id, label) for repo_id, label in PLAYGROUND_REPOS]
    return whoami, checks


def login_if_ready(token: str) -> tuple[dict, list[RepoCheck]]:
    """Validate token + model access; call login() only when everything passes."""
    whoami, checks = check_playground_access(token)
    if not all(c.ok for c in checks):
        return whoami, checks
    login(token=token.strip(), add_to_git_credential=False)
    return whoami, checks
