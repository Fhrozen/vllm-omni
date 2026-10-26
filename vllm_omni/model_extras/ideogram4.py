# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request surface for Ideogram 4: model-specific ``extra_body`` params.

Ideogram 4 is text-to-image only, so a request carries a caption plus the shared
sampling knobs. ``guidance_schedule`` is read in loop-index order (index 0 is the
last, most polished step), matching the reference presets.
"""

from __future__ import annotations

IDEOGRAM4_EXTRA_BODY_PARAMS = frozenset(
    {
        # Which DiT the unconditional CFG branch uses. Startup default comes from
        # ``--custom-pipeline-args`` / ``additional_config`` / the environment; a
        # request-level value only takes effect when both copies are loaded, i.e.
        # when the server was started with ``dit_copies: keep``.
        "dit_copies",
        "guidance_schedule",
        "mu",
        "preset",
        "std",
    }
)
