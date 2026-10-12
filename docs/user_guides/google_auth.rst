Authenticating Gemini backends: Antigravity and OpenCode + Vertex AI
====================================================================

We recommend reaching Gemini through Google credentials that live on the
*host* and are bind-mounted into the LLM worker containers:

* :class:`chia.models.antigravity.AntigravityLLM` — Google's **Antigravity CLI**
  (``agy``). OAuth only; credentials under ``~/.gemini`` by default.
* :class:`chia.models.opencode.OpenCodeLLM` with the ``google-vertex`` provider —
  the **OpenCode CLI** calling Gemini on **Vertex AI**. Google Application
  Default Credentials (ADC) under ``~/.config/gcloud`` by default.

Both can be set up once on the host and then mounted into the credential
directory in the corresponding docker container in your cluster config yaml.
``examples/memcpy/cluster.yaml`` and ``examples/circt_issue_solver/cluster_*.yaml``
 show this.

.. contents::
   :local:
   :depth: 1

Antigravity (``agy``)
---------------------

Install and sign in
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   curl -fsSL https://antigravity.google/cli/install.sh | bash   # installs ~/.local/bin/agy
   agy                                                            # first run: interactive sign-in

``agy`` prints a Google OAuth URL — open it in a browser and sign in with the
Google account whose Antigravity / Gemini Code Assist entitlement you want to
use (on a headless host, copy the URL to your laptop's browser). You'll need to 
pick a **location**: ``global``, ``us`` or ``eu``. The creds get stored in:

* ``~/.gemini/antigravity-cli/antigravity-oauth-token`` — the refresh token
  (there is no API-key path);
* ``~/.gemini/antigravity-cli/settings.json`` — the project and location::

     {"gcp": {"project": "<your-project-id>", "location": "global"}}

Recommendation: **Pick ``global``.** Gemini *Pro* models are served from the global
endpoint; with ``us`` some ``gemini-*-pro*`` requests fails with
*"Selected model is not supported in the selected location"*. You can change it
later by editing ``settings.json`` or by signing in again.

To switch accounts or projects, run ``agy`` interactively and use its ``/logout``
then ``/login`` slash commands (the license selector appears again). 

You confirm what you have:

.. code-block:: bash

   agy models                                          # ids you can pass as model=...
   agy --print "Reply with exactly PONG"               # sanity check
   agy --model gemini-3.1-pro-high --print "Reply PONG"    # Pro reachable? (needs global)
   grep -A3 '"gcp"' ~/.gemini/antigravity-cli/settings.json

The model id suffix (``-high`` / ``-low``) is agy's reasoning-effort tier.

Mounting into the container
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``ghcr.io/ucb-bar/chia-antigravity`` image has ``agy`` installed and reads
its state from ``/home/ray/.gemini``. Bind-mount your signed-in directory there
and run the container as your uid so the mounted files are readable **and
writable** — agy refreshes its token and writes logs and the per-conversation
SQLite databases into ``antigravity-cli/`` on every call. (Chia never edits your
``~/.gemini/config``: each run gets a private ``HOME`` with its own MCP server
list and a symlink to the shared ``antigravity-cli``.)

.. code-block:: yaml

   antigravity:
       resources: {"antigravity_creds": 1}
       docker:
           image: "ghcr.io/ucb-bar/chia-antigravity:latest"
           run_options:
               - "--user $(id -u):$(id -g)"
               - "-v ${HOME}/.gemini:/home/ray/.gemini"
           run_setup_commands:
               - echo "user:x:$(id -u):$(id -g)::/home/ray:/bin/bash" >> /etc/passwd

Notes:

* Recommended to use ``${HOME}`` (or an absolute path), not ``~``
* Mount the whole ``~/.gemini`` tree, not just the token: project/location come
  from ``settings.json`` next to it. Editing ``settings.json`` on the host takes
  effect on the next call inside the container (no restart).
* The ``/etc/passwd`` line gives your uid a home of ``/home/ray`` so ``$HOME``
  resolves correctly inside the container. (``AntigravityLLM`` resolves
  ``~/.gemini`` on the worker that runs the prompt, not where the object was
  built, so it is safe to construct it on a node whose ``HOME`` is ``/root``.)
* The host directory must exist before ``chia up``; Docker creates a missing
  mount source as an empty root-owned directory, which shows up as
  ``mkdir /home/ray/.gemini/antigravity-cli: permission denied``.

Verify from the host after ``chia up``:

.. code-block:: bash

   docker exec <antigravity-container> agy --print "Reply with exactly PONG"

If it prints an OAuth URL instead, the mounted directory is not the signed-in
one (or is empty).

OpenCode with Gemini on Vertex AI
---------------------------------

OpenCode is provider-agnostic — the model is ``<provider>/<model>`` and each
provider brings its own credentials. For Gemini on Vertex AI the provider is
``google-vertex``, which needs two things: **Google Application Default
Credentials (ADC)** and a **GCP project**.
Setup is three steps: sign in on the host, mount + configure the container, and
tell the driver which model/project to use.

1. Host: sign in and pick the project
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   gcloud auth application-default login                        # writes ~/.config/gcloud/application_default_credentials.json
   gcloud auth application-default set-quota-project <project>  # required for user credentials, else 403
   gcloud services enable aiplatform.googleapis.com --project <project>

Check the identity you just created — ADC is a **separate credential from the
gcloud CLI account** (``gcloud auth list`` can show a different user):

.. code-block:: bash

   curl -s "https://www.googleapis.com/oauth2/v3/tokeninfo?access_token=$(gcloud auth application-default print-access-token)" | grep email

That identity needs the *Vertex AI User* role on ``<project>``. For unattended
hosts you can use a service-account key file instead of a login (step 2 shows
how to point at it).

2. Container: mount the credentials, set project + location
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``ghcr.io/ucb-bar/chia-opencode`` image needs no ``gcloud``: OpenCode's Google
auth library reads the ADC file directly. Mount the gcloud config directory and
pass the project/location as environment variables — the ``google-vertex``
provider reads them from the ``opencode`` process's environment, not from
``gcloud config``:

.. code-block:: yaml

   opencode:
       resources: {"opencode_creds": 1}
       docker:
           image: "ghcr.io/ucb-bar/chia-opencode:latest"
           run_options:
               - "-v ${HOME}/.config/gcloud:/home/ray/.config/gcloud"   # ADC file
               - "-e GOOGLE_CLOUD_PROJECT=${GOOGLE_CLOUD_PROJECT}"     # project
               - "-e VERTEX_LOCATION=global"                           # Gemini Pro is global-only
               # service-account key instead of a login:
               # - "-v /path/to/sa-key.json:/home/ray/sa-key.json:ro"
               # - "-e GOOGLE_APPLICATION_CREDENTIALS=/home/ray/sa-key.json"

``export GOOGLE_CLOUD_PROJECT=<project>`` **before** ``chia up`` so the
``${GOOGLE_CLOUD_PROJECT}`` substitution has a value, and make sure
``~/.config/gcloud`` exists (Docker would otherwise create an empty root-owned
directory). The mount is read-only in practice, so no uid tricks are needed.

Verify from the host:

.. code-block:: bash

   docker exec <opencode-container> opencode run -m google-vertex/gemini-3.1-pro-preview "Reply with exactly PONG"

3. Driver: choose the model and pass the project through
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In CHIA code, select a Vertex model as ``google-vertex/<vertex model id>`` and
pin the project/location in OpenCode's config with an ``AdditionalModelProvider``
(this also registers the model id in case OpenCode's catalog lags Vertex)::

   from chia.models.opencode import OpenCodeLLM, AdditionalModelProvider

   vertex = AdditionalModelProvider(
       id="google-vertex", npm="@ai-sdk/google-vertex", name="Google Vertex AI",
       models=["gemini-3.1-pro-preview"],
       options={"project": "<project>", "location": "global"},
   )
   llm = OpenCodeLLM(model="google-vertex/gemini-3.1-pro-preview",
                     additional_providers=[vertex])

Config values take precedence over the container's ``GOOGLE_CLOUD_PROJECT`` /
``VERTEX_LOCATION``; those env vars remain as a fallback (e.g. for the manual
``opencode run`` check above). Credentials are never in the config — they always
come from the mounted ADC.

The memcpy example does this (``examples/memcpy/llm.py``) whenever
``MEMCPY_OPENCODE_MODEL`` starts with ``google-vertex/``, reading the project from
``GOOGLE_CLOUD_PROJECT`` on the **driver**. A ``chia job submit`` driver does not
inherit your shell, so forward both values through the job's runtime env:

.. code-block:: bash

   chia job submit \
     --runtime-env-json "{\"env_vars\": {\"MEMCPY_OPENCODE_MODEL\": \"google-vertex/gemini-3.1-pro-preview\", \"GOOGLE_CLOUD_PROJECT\": \"$GOOGLE_CLOUD_PROJECT\"}}" \
     -- python $PWD/examples/memcpy/memcpy_loop.py --llm opencode

Quick reference
---------------

.. list-table::
   :header-rows: 1
   :widths: 18 41 41

   * -
     - Antigravity (``agy``)
     - OpenCode + Vertex AI
   * - Sign in
     - ``agy`` (interactive OAuth; ``/logout`` + ``/login`` to switch)
     - ``gcloud auth application-default login`` + ``set-quota-project``
   * - Project / location
     - chosen at sign-in; ``~/.gemini/antigravity-cli/settings.json``
     - ``GOOGLE_CLOUD_PROJECT`` / ``VERTEX_LOCATION`` (+ pinned via ``AdditionalModelProvider``)
   * - Credential on disk
     - ``~/.gemini/antigravity-cli/antigravity-oauth-token``
     - ``~/.config/gcloud/application_default_credentials.json`` (or a SA key)
   * - Mount
     - ``-v ${HOME}/.gemini:/home/ray/.gemini`` (+ ``--user``, writable)
     - ``-v ${HOME}/.config/gcloud:/home/ray/.config/gcloud`` (read-only is fine)
   * - Gemini Pro
     - location ``global``
     - ``VERTEX_LOCATION=global``
   * - Model id
     - ``agy models`` (e.g. ``gemini-3.1-pro-high``)
     - ``google-vertex/<vertex id>`` (e.g. ``gemini-3.1-pro-preview``)
