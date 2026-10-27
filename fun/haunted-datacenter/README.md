# Haunted Datacenter Escape Room

Run a three-room escape game that combines web research, vision inference, LoRA fine-tuning, and a cloud cooling controller. Two or three people discuss the puzzles while one person operates the local browser. The haunted building and sensor readings are simulated; the API calls, training job, and Room 3 deployment are real.

This recipe uses [Haunted Datacenter by gogococo](https://github.com/gogococo/haunted-datacenter), release [`stream-host-v1`](https://github.com/gogococo/haunted-datacenter/releases/tag/stream-host-v1), commit `0c0611cdabf64c48bd60074f24082e0f12d39781`. The game code stays in its upstream repository. Later releases have a different training and hosting architecture.

## What You Will Build

The local game server coordinates these services:

| Room | Task | Service that does real work |
| --- | --- | --- |
| Haunted switchboard | Research MQTT wildcard rules, inspect a source, and repair message routing | Tavily Search and Extract |
| Network operations center | Inspect camera images, record demonstrations, and fine-tune the caretaker | Nebius Token Factory vision and managed LoRA fine-tuning |
| Generator room | Repair the simulated loop, deploy the controller, and verify cooling | Nebius AI Cloud Serverless AI CPU endpoint |

The trained caretaker runs locally using `Qwen/Qwen3-0.6B` and the adapter from your own Token Factory job. This release does not require a warm training GPU or a new Token Factory text-serving endpoint. Training opens the maintenance hatch so you can work on Room 3 while the job runs, then load and assess the checkpoint at the portable terminal.

## Credentials and Access

Prepare the following before starting. Local hosting still calls paid cloud services.

| Credential or setting | Required locally? | Purpose and where to obtain it |
| --- | --- | --- |
| `NEBIUS_API_KEY` | Yes | Token Factory API key from the **API keys** section of [Token Factory](https://tokenfactory.nebius.com/). The account needs access to vision inference, file uploads, fine-tuning for `Qwen/Qwen3-0.6B`, and checkpoint retrieval. The game also lists dedicated endpoints and templates during connection. |
| `TAVILY_API_KEY` | Yes, or an authenticated Tavily CLI session | Key from [Tavily](https://app.tavily.com/) for Room 1 Search and Extract. `tvly login` is an alternative to the environment variable. |
| Nebius AI Cloud CLI authentication | Yes | A signed-in `nebius` CLI profile with permission to inspect the selected project/subnet and create, inspect, read logs from, and stop Serverless AI endpoints. Use your own [Nebius AI Cloud account](https://console.nebius.com/). This is separate from the Token Factory API key. |
| `NEBIUS_PROJECT_ID` | Yes; identifier, not a secret | Your AI Cloud project with billing and available Serverless CPU quota. |
| `NEBIUS_SUBNET_ID` | Yes; identifier, not a secret | A subnet in that project and a region supporting the controller's `cpu-d3` / `2vcpu-8gb` configuration. |
| Vision model access | Yes; no additional key if covered by `NEBIUS_API_KEY` | An image-capable serverless model visible to this account, or an enabled compatible dedicated vision endpoint. The game's **Connect Token Factory** action discovers these. Test a camera image before the event. |
| Hugging Face token | Usually no | The public Qwen base model can be downloaded anonymously. Use `hf auth login` only if your environment requires authentication. |
| `HOST_ACCESS_CODE`, `SESSION_SECRET` | No | Used for optional host sign-in/public hosting. Leave unset for this local recipe. |
| `NEBIUS_SERVICE_ACCOUNT_ID`, `NEBIUS_PUBLIC_KEY_ID`, `NEBIUS_PRIVATE_KEY` | No, when using an interactive CLI profile | Alternative unattended authentication for containers. A service-account private key is not needed for the local interactive setup. |

Follow the official [Token Factory authentication guide](https://docs.tokenfactory.nebius.com/api-reference/introduction) and check [fine-tuning model support](https://docs.tokenfactory.nebius.com/post-training/models) for your account. Model availability, quota, and access can change; a successful model listing alone does not prove that a training job will be accepted.

Keep keys in the server environment or a private `.env` file. Never put them in browser fields, chat prompts, screenshots, or Git. The game creates its local runner control token and cloud controller bearer token itself; you do not need to obtain those separately.

## Set Up Locally

Use Git, Python 3.12 with TLS support, the Nebius AI Cloud CLI, and enough disk and memory for the public 0.6B base model and Python ML dependencies. Use separate terminals for the model runner and game server. No Docker, domain, registry, public web hosting, or cloud GPU reservation is required for this path.

### 1. Check out the release

Clone into a separate working directory outside the cookbook checkout. This avoids adding game runtime files and model weights to the cookbook.

```bash
git clone --branch stream-host-v1 --depth 1 https://github.com/gogococo/haunted-datacenter.git
cd haunted-datacenter
git rev-parse HEAD
```

Confirm the printed SHA is `0c0611cdabf64c48bd60074f24082e0f12d39781`. Stop if it differs.

### 2. Install the local tools and model

Use the release's dependency pins. Install a platform-compatible PyTorch build using the [official PyTorch installer](https://pytorch.org/get-started/locally/); the following command uses pip's default platform selection.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch
python -m pip install -r requirements-model.txt
hf download Qwen/Qwen3-0.6B \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --local-dir models/base
tvly --status
nebius ai endpoint create --help
```

The model revision matches the release's Dockerfile. `requirements-model.txt` includes the Tavily CLI. If the pinned packages are unavailable for your Python/platform, resolve and record that compatibility issue before continuing; do not silently replace the release's pins. The tagged Dockerfile uses Nebius CLI `0.12.295`; check that your CLI supports the endpoint flags in `serverless.py`, including file injection.

Authenticate the Nebius CLI using its installed version's setup instructions. Check project/subnet access in the console and configure the selected profile before launching the game. The game invokes this CLI noninteractively, so browser login must be completed in advance.

### 3. Configure the game server

Copy this recipe's [`env.example`](env.example) into the game checkout as `.env`. Fill the four required values locally using an editor. The game does not load `.env` automatically.

```bash
chmod 600 .env
set -a
source .env
set +a
```

Load only a file you created and trust: `source` executes shell syntax. Keep this shell open for the game server. If you prefer Tavily OAuth, leave `TAVILY_API_KEY` blank and run `tvly login` before starting.

### 4. Start the model runner

In the first terminal, run from the game checkout:

```bash
source .venv/bin/activate
python caretaker_runner.py --base-dir models/base
```

The runner listens on `127.0.0.1:8770` and starts with base weights. It uses MPS when available, otherwise CPU. Wait for the model to load before opening the game.

### 5. Start the game

In the terminal that loaded `.env`, run:

```bash
source .venv/bin/activate
python server.py
```

Open **http://127.0.0.1:8765**, enter a display name, and select **Play**. This tag has one shared incident controlled from one keyboard. Refreshing resumes that incident; separate tabs do not create independent player saves.

Keep the default `.runtime/` data directory for both processes. If you change `GAME_DATA_DIR`, also pass that exact absolute path to the runner's `--runtime` option so checkpoint loading uses the same private receipts.

## One-Shot Setup Prompt

Paste the following into a coding agent with terminal access, or use the standalone [setup prompt](setup-prompt.txt). It prepares the local environment and checks cloud access. Interactive account login, missing credentials, quota changes, and unavailable models can still need your input.

```text
Set up Haunted Datacenter locally from gogococo/haunted-datacenter release
stream-host-v1, exact commit 0c0611cdabf64c48bd60074f24082e0f12d39781.
Read its tagged README, serverless.py, caretaker.py, and dependency files first.
Use this release's managed Token Factory fine-tuning and local model runner.

1. Clone into a separate directory, verify the commit, and use Python 3.12
   with an isolated virtual environment. Install platform-compatible PyTorch
   and the release's requirements-model.txt. Download Qwen/Qwen3-0.6B at
   c1899de289a04d12100db370d81485cdf75e47ca using hf into models/base.
   Report dependency conflicts; do not silently change pins or game code.
2. Check for NEBIUS_API_KEY and Tavily CLI authentication (TAVILY_API_KEY or
   tvly login), without printing credential values. Use a private ignored
   .env if needed and explicitly load it; this game has no dotenv loader.
   Ask me to enter missing secrets locally, never in chat. Check the nebius
   CLI profile is already authenticated. A Token Factory key cannot replace it.
3. Discover/select my AI Cloud project and subnet using read-only operations.
   Ask me to select if multiple plausible projects exist. Check available
   quota and compatibility for one cpu-d3 / 2vcpu-8gb controller with 20 GiB
   disk. Export NEBIUS_PROJECT_ID and NEBIUS_SUBNET_ID to the game process.
   Reuse my project/subnet. If either is missing, explain the exact required
   resource and request the missing selection or creation authorization.
4. Check Token Factory account access to model listing, dedicated endpoint
   listing/templates, Qwen/Qwen3-0.6B fine-tuning, and checkpoint retrieval.
   Distinguish read-only checks from capabilities only a paid job can prove.
   Prefer a compatible image-capable serverless model already available to
   the account. If a dedicated vision endpoint is necessary, first report
   its model, region, capacity, and billing implications and get approval.
   Reuse a compatible enabled endpoint instead of creating a duplicate.
5. Start the local runner on 127.0.0.1:8770 and the game on 127.0.0.1:8765.
   Keep both on loopback and use the same private runtime directory. Record
   process IDs and log paths without copying secrets into the summary.
   Verify runner /health, game /healthz, and the welcome page. Run the tagged
   automated tests in a temporary GAME_DATA_DIR. Explain any failed checks.
6. Leave training submission and the Room 3 cooling deployment to the player.
   Do not submit a training job, create a cooling endpoint, pre-solve rooms,
   or load an old adapter during setup. Do not create a cloud game host,
   warm GPU worker, text-serving endpoint, registry, or public tunnel.
7. Return the local URL, required credential names and readiness (no values),
   selected project/subnet and vision model, validation evidence, unresolved
   prerequisites, and cleanup instructions. List any resources created with
   their IDs. Explain that API calls, fine-tuning, and cloud endpoints bill
   separately, and resets do not cancel jobs or stop endpoints in this tag.
```

## Verify and Play

Check local readiness from another terminal:

```bash
curl --fail http://127.0.0.1:8770/health
curl --fail http://127.0.0.1:8765/healthz
```

The runner should report ready with base weights. In the NOC deployment rack, **Connect Token Factory** should discover account resources and the local runner. Inspect a camera to verify real image inference. This inference is billed; the local health checks are not provider tests.

Run the tagged automated tests with disposable state:

```bash
GAME_DATA_DIR=$(mktemp -d) .venv/bin/python -m unittest -q \
  test_game test_caretaker test_plant test_upgrade test_checkpoint_flow test_hosting
```

These tests mock external services. A full integration check requires completing Room 1 Search and Extract, submitting your own real fine-tuning job, loading its adapter, passing the fresh camera assessment, and passing the deployed controller's cooling check. Preserve the job/endpoint IDs for cleanup. Do not describe mocked tests as a verified cloud playthrough.

For checkpoint-download `AccessDenied`, use the game's explicit import flow for the final `adapter_model.safetensors` from that same job. Selecting an older adapter is a host override, not completion of the default player path. See the [tagged game instructions](https://github.com/gogococo/haunted-datacenter/blob/0c0611cdabf64c48bd60074f24082e0f12d39781/README.md) for the full room mechanics.

## Costs and Cleanup

Tavily Search/Extract, Token Factory inference and fine-tuning, and the cloud CPU endpoint have separate usage charges. Check your account pricing and budget before a playthrough. Cloud provisioning and training can take several minutes; the 45-minute format is presentation pacing, not a completion-time guarantee.

This tag preserves cloud resources across game resets. Before resetting or removing local state, record all created endpoint and training job IDs from the private runtime receipts or provider consoles. Inspect current and earlier attempts; an uncertain create can have succeeded even when the game did not receive a response.

Stop the local game and runner with Ctrl-C. In Nebius AI Cloud, stop the cooling endpoints created for your playthrough and verify their state. Stop any dedicated vision endpoint created solely for this demo through Token Factory. Check for unfinished training jobs and cancel them if no longer needed. Review any retained storage before deleting it. Preserve `.runtime/` if you need receipts, saves, adapters, or scores; keep it private and out of Git.

## Troubleshooting

Use these checks when setup fails:

| Symptom | Check |
| --- | --- |
| Token Factory works but Room 3 cannot deploy | The `nebius` CLI profile, project/subnet, endpoint permissions, compatible CLI flags, and CPU quota. |
| No camera model appears | Account-visible image models or an enabled dedicated vision endpoint; Connect Token Factory must succeed. |
| Tavily unavailable | Activate the virtual environment before starting the server; `tvly` must be on its PATH and authenticated. |
| Runner missing or checkpoint cannot load | Port 8770, matching runtime directory, exact base model, and adapter from this playthrough's job. |
| Missing credentials despite `.env` being present | Load `.env` into the server's shell before launching it. |
| Cloud create timed out | Check the provider console and receipts before retrying to avoid duplicate resources. |
