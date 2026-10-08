"""Resolve ``rucio://scope:name`` sample entries into ``root://`` file URLs.

Talks to the Rucio REST API directly with ``requests`` and the user's grid
proxy, so no Rucio client installation (or ``lsetup rucio``) is needed. The
defaults point at the ATLAS Rucio servers; override them with the
``RUCIO_HOST`` / ``RUCIO_AUTH_HOST`` environment variables.

Environment:
  X509_USER_PROXY   grid proxy (default /tmp/x509up_u<uid>), from
                    ``voms-proxy-init -voms atlas``
  X509_CERT_DIR     CA certificates (default /etc/grid-security/certificates)
  RUCIO_ACCOUNT     Rucio account; optional if the proxy maps to one account
"""
from __future__ import annotations

import json
import os
from typing import Optional

RUCIO_PREFIX = "rucio://"
DEFAULT_RUCIO_HOST = "https://voatlasrucio-server-prod.cern.ch"
DEFAULT_RUCIO_AUTH_HOST = "https://voatlasrucio-auth-prod.cern.ch"


def is_rucio_did(entry) -> bool:
    return isinstance(entry, str) and entry.startswith(RUCIO_PREFIX)


def parse_did(entry: str) -> tuple[str, str]:
    """``rucio://scope:name`` -> (scope, name)."""
    did = entry[len(RUCIO_PREFIX):]
    scope, sep, name = did.partition(":")
    if not sep or not scope or not name:
        raise ValueError(f"Rucio entries must look like rucio://<scope>:<name>, got: {entry!r}")
    return scope, name


def _proxy_path() -> str:
    path = os.environ.get("X509_USER_PROXY", f"/tmp/x509up_u{os.getuid()}")
    if not os.path.isfile(path):
        raise RuntimeError(
            f"No grid proxy found at {path}. Run `voms-proxy-init -voms atlas` "
            "(or set X509_USER_PROXY) before using rucio:// inputs."
        )
    return path


def _ca_verify():
    ca_dir = os.environ.get("X509_CERT_DIR", "/etc/grid-security/certificates")
    return ca_dir if os.path.isdir(ca_dir) else True


class RucioResolver:
    """Resolves Rucio DIDs to ``root://`` PFNs, authenticating once per instance."""

    def __init__(self, preferred_rses: Optional[list[str]] = None):
        import requests  # only needed when rucio:// inputs are used

        self.preferred_rses = list(preferred_rses or [])
        self.host = os.environ.get("RUCIO_HOST", DEFAULT_RUCIO_HOST).rstrip("/")
        self.auth_host = os.environ.get("RUCIO_AUTH_HOST", DEFAULT_RUCIO_AUTH_HOST).rstrip("/")
        self.session = requests.Session()
        self.session.verify = _ca_verify()
        self._token = None
        self._resolved = {}

    def _auth_token(self) -> str:
        if self._token is None:
            proxy = _proxy_path()
            headers = {}
            account = os.environ.get("RUCIO_ACCOUNT")
            if account:
                headers["X-Rucio-Account"] = account
            resp = self.session.get(
                f"{self.auth_host}/auth/x509_proxy", headers=headers, cert=(proxy, proxy)
            )
            if resp.status_code != 200 or "X-Rucio-Auth-Token" not in resp.headers:
                hint = "" if account else " If your proxy maps to several accounts, set RUCIO_ACCOUNT."
                raise RuntimeError(
                    f"Rucio authentication failed (HTTP {resp.status_code}): {resp.text.strip()[:300]}.{hint}"
                )
            self._token = resp.headers["X-Rucio-Auth-Token"]
        return self._token

    def _list_replicas(self, scope: str, name: str) -> list[dict]:
        resp = self.session.post(
            f"{self.host}/replicas/list",
            headers={
                "X-Rucio-Auth-Token": self._auth_token(),
                "Accept": "application/x-json-stream",
                "Content-Type": "application/json",
            },
            data=json.dumps({"dids": [{"scope": scope, "name": name}], "schemes": ["root"]}),
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"Rucio replica lookup for {scope}:{name} failed (HTTP {resp.status_code}): "
                f"{resp.text.strip()[:300]}"
            )
        return [json.loads(line) for line in resp.text.splitlines() if line.strip()]

    def ranked_pfns(self, replica: dict) -> list[tuple[str, str]]:
        """All usable (rse, pfn) for a file, best first: preferred RSEs in the
        given order, then the remaining disk replicas by Rucio priority."""
        candidates = []
        for pfn, info in (replica.get("pfns") or {}).items():
            if not pfn.startswith("root://") or info.get("type") == "TAPE":
                continue
            rse = info.get("rse", "")
            if replica.get("states", {}).get(rse, "AVAILABLE") != "AVAILABLE":
                continue
            candidates.append((rse, pfn, info.get("priority", 0)))
        rank = {rse: i for i, rse in enumerate(self.preferred_rses)}
        candidates.sort(key=lambda c: (rank.get(c[0], len(rank)), c[2]))
        return [(rse, pfn) for rse, pfn, _ in candidates]

    def resolve(self, entry: str) -> list[str]:
        """``rucio://scope:name`` (file, dataset or container) -> sorted root:// PFNs."""
        if entry not in self._resolved:
            self._resolved[entry] = self._resolve(entry)
        return self._resolved[entry]

    def _resolve(self, entry: str) -> list[str]:
        scope, name = parse_did(entry)
        replicas = self._list_replicas(scope, name)
        if not replicas:
            raise RuntimeError(f"Rucio found no files for {scope}:{name}")
        chosen, missing, rses = [], [], set()
        for replica in replicas:
            ranked = self.ranked_pfns(replica)
            if not ranked:
                missing.append(f"{replica.get('scope')}:{replica.get('name')}")
                continue
            rses.add(ranked[0][0])
            chosen.append((replica.get("name", ""), ranked[0][1]))
            _ALTERNATIVES[ranked[0][1]] = [pfn for _, pfn in ranked[1:]]
            _DATASET_OF[ranked[0][1]] = name
        if missing:
            raise RuntimeError(
                f"{len(missing)} of {len(replicas)} files in {scope}:{name} have no available "
                f"disk replica reachable via root://, e.g. {missing[0]}"
            )
        print(f"{entry}: {len(chosen)} files from {', '.join(sorted(rses))}")
        return [pfn for _, pfn in sorted(chosen)]


# Primary PFN -> the file's other replicas, best first, so a reader can fall
# back to another site if the first one does not respond.
_ALTERNATIVES: dict[str, list[str]] = {}


# Primary PFN -> name of the Rucio dataset/container it was resolved from
_DATASET_OF: dict[str, str] = {}


def dataset_of(url: str) -> Optional[str]:
    """Rucio dataset name a file was resolved from, or None for other inputs."""
    return _DATASET_OF.get(url)


def replica_alternatives(url: str) -> list[str]:
    """Other replicas of a file resolved from a rucio:// entry (empty otherwise)."""
    return list(_ALTERNATIVES.get(url, []))


# One resolver per site preference, so a run authenticates once and looks up
# each dataset once even if it is listed as both signal and background.
_RESOLVERS: dict[tuple, RucioResolver] = {}


def expand_rucio_samples(samples: list, preferred_rses: Optional[list[str]] = None) -> list:
    """Replace each ``rucio://`` entry (alone, or inside a multi-file sample)
    with its file URLs; other entries are left unchanged. One sample stays one
    sample, so per-sample weights still apply to the whole dataset."""
    if not any(
        is_rucio_did(f) for s in samples for f in (s if isinstance(s, (list, tuple)) else [s])
    ):
        return samples
    key = tuple(preferred_rses or [])
    if key not in _RESOLVERS:
        _RESOLVERS[key] = RucioResolver(preferred_rses)
    resolver = _RESOLVERS[key]
    expanded = []
    for sample in samples:
        files = sample if isinstance(sample, (list, tuple)) else [sample]
        out = []
        for f in files:
            out.extend(resolver.resolve(f) if is_rucio_did(f) else [f])
        expanded.append(out if (isinstance(sample, (list, tuple)) or is_rucio_did(sample)) else sample)
    return expanded
