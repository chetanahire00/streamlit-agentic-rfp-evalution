"""Left sidebar: AI provider configuration, connection test and advanced limits.

The API key lives only in ``st.session_state`` (the password widget's value). It
is never written to SQLite, logs, exports or messages; only its masked form is shown.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import streamlit as st

from config.settings import Settings
from src.llm.base import LLMConfig
from src.llm.catalog import ProviderOption
from src.llm.factory import create_provider
from src.ui.components import badge_html, render_html
from src.ui.state import CONN_STATUS
from src.utils.errors import LLMError
from src.utils.logging_setup import get_logger
from src.utils.secrets import mask_secret, register_secret

log = get_logger(__name__)

CUSTOM_MODEL = "Custom model ID…"


@dataclass(frozen=True)
class SidebarState:
    """What the rest of the app needs from the sidebar."""

    config: LLMConfig | None
    key_source: str
    connection_ok: bool
    skip_test: bool
    max_pdf_chars: int
    problems: list[str]

    @property
    def llm_ready(self) -> bool:
        """Provider, model and key present, and connection tested (or explicitly skipped)."""
        return self.config is not None and not self.problems and (self.connection_ok or self.skip_test)


def resolve_api_key(typed: str, secret_name: str | None) -> tuple[str, str]:
    """Pick the key from the sidebar input, then ``st.secrets``, then the environment.

    Returns:
        ``(key, source_label)``. The key is ``""`` when none is found.
    """
    if typed.strip():
        return typed.strip(), "Sidebar input"
    if not secret_name:
        return "", "Not required"
    try:
        value = st.secrets.get(secret_name, "")
    except Exception:  # noqa: BLE001 - no secrets.toml configured
        value = ""
    if value:
        return str(value).strip(), f"st.secrets[{secret_name}]"
    env = os.environ.get(secret_name, "")
    if env:
        return env.strip(), f"Environment variable {secret_name}"
    return "", "Missing"


def _signature(cfg: LLMConfig) -> str:
    raw = f"{cfg.provider}|{cfg.model}|{cfg.api_key}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def render_sidebar(catalog: list[ProviderOption], settings: Settings) -> SidebarState:
    """Draw the sidebar and return the resulting configuration."""
    with st.sidebar:
        st.markdown("### :material/smart_toy: AI provider")
        labels = {p.label: p for p in catalog}
        provider = labels[st.selectbox("AI Provider", list(labels), key="sb_provider",
                                       help="Which LLM service scores the proposals.")]

        model_labels = [f"{m.label} · {m.id}" for m in provider.models] + [CUSTOM_MODEL]
        choice = st.selectbox("Model", model_labels, key=f"sb_model_{provider.id}",
                              help="Model list from config/providers.yaml.")
        if choice == CUSTOM_MODEL:
            model_id = st.text_input("Custom model ID", key=f"sb_custom_{provider.id}",
                                     placeholder="exact model id from the provider docs").strip()
        else:
            model_id = provider.models[model_labels.index(choice)].id

        problems: list[str] = []
        key, source = "", "Not required"
        if provider.requires_key:
            typed = st.text_input("API Key", type="password", key=f"sb_key_{provider.id}",
                                  help=f"Kept only in this browser session. Leave empty to use "
                                       f"st.secrets['{provider.secret_name}'].",
                                  autocomplete="off")
            key, source = resolve_api_key(typed, provider.secret_name)
            register_secret(key)
            if key:
                st.caption(f"Key source: **{source}** · `{mask_secret(key)}`")
            else:
                st.caption(f"No key found. Enter one or set `{provider.secret_name}` in secrets.")
                problems.append("Enter an API key")
        else:
            st.caption("Runs offline with deterministic keyword scoring. No key needed; "
                       "not an AI model.")
        if not model_id:
            problems.append("Choose a model")

        with st.expander("Advanced", icon=":material/tune:"):
            lock = st.checkbox("Lock temperature to 0 (reproducible)", value=True, key="sb_lock_temp")
            temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.05, disabled=lock, key="sb_temp")
            if lock:
                temperature = 0.0
            if not provider.supports_temperature:
                st.caption("This provider does not accept a temperature for these models; repeat "
                           "runs are kept identical by the result cache.")
            max_tokens = int(st.number_input("Max output tokens", 512, 64_000, settings.default_max_tokens,
                                             256, key="sb_max_tokens"))
            timeout = float(st.number_input("Request timeout (s)", 5, 600, int(settings.default_timeout_s),
                                            5, key="sb_timeout"))
            retries = int(st.number_input("Max retries", 0, 5, settings.default_max_retries, 1,
                                          key="sb_retries"))
            max_chars = int(st.number_input("Max PDF characters sent to the LLM", 2_000, 400_000,
                                            settings.max_pdf_chars, 1_000, key="sb_max_chars"))

        config = None
        if not problems:
            config = LLMConfig(provider=provider.id, model=model_id, api_key=key, temperature=temperature,
                               max_tokens=max_tokens, timeout_s=timeout, max_retries=retries,
                               supports_temperature=provider.supports_temperature,
                               provider_label=provider.label)

        connection_ok = _connection_block(config)
        skip = st.checkbox("Skip connection test", key="sb_skip_test",
                           help="Allow evaluation without a successful test (not recommended).")
        if config is None:
            render_html(badge_html("✕ " + "; ".join(problems), "danger"))
    return SidebarState(config, source, connection_ok, skip, max_chars, problems)


def _connection_block(config: LLMConfig | None) -> bool:
    status = st.session_state.get(CONN_STATUS)
    signature = _signature(config) if config else None
    if status and status.get("signature") != signature:
        status = None  # provider, model or key changed: the old result no longer applies
    clicked = st.button("Test connection", icon=":material/network_check:", disabled=config is None,
                        width="stretch", key="sb_test")
    if clicked and config is not None:
        with st.spinner(f"Contacting {config.label}…"):
            try:
                create_provider(config).test_connection()
                status = {"signature": signature, "ok": True, "message": f"Connected to {config.label}"}
            except LLMError as exc:
                status = {"signature": signature, "ok": False, "message": exc.user_message}
            except Exception:
                log.exception("Connection test crashed")
                status = {"signature": signature, "ok": False,
                          "message": "Unexpected error during the connection test."}
        st.session_state[CONN_STATUS] = status
    if status:
        tone, icon = ("success", "✓") if status["ok"] else ("danger", "✕")
        render_html(badge_html(f"{icon} {status['message']}", tone))
        return bool(status["ok"])
    if config is not None:
        render_html(badge_html("○ Not tested yet", "neutral"))
    return False
