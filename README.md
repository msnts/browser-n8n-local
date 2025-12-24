# Browser Use Local Bridge for n8n

![Docker Hub Publish](https://github.com/msnts/browser-n8n-local/actions/workflows/docker-publish.yml/badge.svg)

This is a local bridge service that enables n8n to communicate with the Browser Use Python library. It mimics the Browser Use Cloud API endpoints but runs locally, allowing you to execute browser automation tasks without relying on the cloud service.

## Features

- Compatible with the Browser Use Cloud API endpoints
- Supports both OpenAI and Anthropic language models
- Provides task management (run, pause, resume, stop)
- Exposes status tracking and result retrieval

## Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- Browser Use Python library
- API keys for OpenAI or Anthropic (depending on which LLM you want to use)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/henry0hai/browser-n8n-local.git
   cd browser-n8n-local
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env-example .env
   ```
   Then edit the `.env` file to add your OpenAI and/or Anthropic API keys.

## Running the Service

1. Start the FastAPI server:
   ```bash
   python app.py
   ```

2. The server will start at http://localhost:8000 by default.

3. You can access the API documentation at http://localhost:8000/docs

## API Endpoints

| Method | Endpoint                           | Description                  |
|--------|------------------------------------|------------------------------|
| POST   | /api/v1/run-task                   | Start a new browser task     |
| GET    | /api/v1/task/{task_id}             | Get task details             |
| GET    | /api/v1/task/{task_id}/status      | Get task status              |
| PUT    | /api/v1/stop-task/{task_id}        | Stop a running task          |
| PUT    | /api/v1/pause-task/{task_id}       | Pause a running task         |
| PUT    | /api/v1/resume-task/{task_id}      | Resume a paused task         |
| GET    | /api/v1/list-tasks                 | List all tasks               |
| GET    | /live/{task_id}                    | Live view UI                 |
| GET    | /api/v1/ping                       | Check health                 |
| GET    | /api/v1/task/{task_id}/media       | Get task media               |
| GET    | /api/v1/task/{task_id}/media/list  | List all media from task     |
| GET    | /api/v1/media/{task_id}/{filename} | Display task media content   |

## Usage Examples

### Starting a Task

```bash
curl -X POST http://localhost:8000/api/v1/run-task \
  -H "Content-Type: application/json" \
  -d '{"task": "Go to google.com and search for n8n automation", "ai_provider": "openai"}'
```

### Checking Task Status

```bash
curl -X GET http://localhost:8000/api/v1/task/{task_id}/status
```

### Stopping a Task

```bash
curl -X PUT http://localhost:8000/api/v1/stop-task/{task_id}
```

## Configuration Options

You can configure the service by editing the `.env` file.  Available options are grouped below:

### API Configuration

- `PORT`: The port the service will run on (default: 8000).

### LLM Provider Configuration

The application supports multiple AI providers. You can specify the provider in each request using the `ai_provider` parameter. If not specified, it defaults to `openai`. To change the default provider, set the `DEFAULT_AI_PROVIDER` environment variable.

#### OpenAI

- `OPENAI_API_KEY`: Your OpenAI API key.
- `OPENAI_MODEL_ID`: The model to use (e.g., `gpt-4o`).
- `OPENAI_BASE_URL`: Optional custom endpoint for OpenAI compatible APIs.

#### Anthropic

- `ANTHROPIC_API_KEY`: Your Anthropic API key.
- `ANTHROPIC_MODEL_ID`: The model to use (e.g., `claude-3-opus-20240229`).

#### MistralAI

- `MISTRAL_API_KEY`: Your MistralAI API key.
- `MISTRAL_MODEL_ID`: The model to use (e.g., `mistral-large-latest`).

#### Google AI

- `GOOGLE_API_KEY`: Your Google AI API key.
- `GOOGLE_MODEL_ID`: The model to use (e.g., `gemini-1.5-pro`).

#### Ollama

- `OLLAMA_API_BASE`: The base URL for your Ollama instance.
- `OLLAMA_MODEL_ID`: The model to use (e.g., `llama3`).

#### Azure OpenAI

- `AZURE_API_KEY`: Your Azure OpenAI API key.
- `AZURE_ENDPOINT`: Your Azure OpenAI endpoint URL.
- `AZURE_DEPLOYMENT_NAME`: Your deployment name.
- `AZURE_API_VERSION`: API version to use.

#### Amazon Bedrock

- `BEDROCK_MODEL_ID`: The model ID to use for Amazon Bedrock (e.g., `anthropic.claude-3-sonnet-20240229-v1:0`).
- `AWS_ACCESS_KEY_ID`: Your AWS Access Key ID.
- `AWS_SECRET_ACCESS_KEY`: Your AWS Secret Access Key.
- `AWS_REGION`: The AWS region where your Bedrock service is hosted (e.g., `us-east-1`).

If `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_REGION` are not explicitly set, the AWS SDK will attempt to use its default credential provider chain (e.g., IAM roles, shared credentials file).

### Optional Configuration

- `LOG_LEVEL`: Logging level (default: `INFO`).
- `BROWSER_USE_HEADFUL`: Set to `"true"` to run the browser in headful mode (default: `false`, runs in headless mode).

## Troubleshooting

- **ImportError with browser-use**: Make sure you have installed the browser-use package and its dependencies correctly.
- **API Key Issues**: Verify that your API keys are correctly set in the `.env` file.
- **Port Conflicts**: If port 8000 is already in use, set a different port in the `.env` file.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Browser Use](https://github.com/browser-use/browser-use) - The underlying browser automation library
- [FastAPI](https://fastapi.tiangolo.com/) - The web framework used
- [n8n](https://n8n.io/) - The workflow automation platform this bridge is designed for # browser-n8n-local

**CI / CD: Docker image publishing**

- **What the workflow does:** Builds a multi-arch Docker image (amd64 + arm64) and pushes to GitHub Container Registry (`ghcr.io`) and Docker Hub (optional) when commits are pushed to `main` or tags starting with `v` or `release-`.
- **Workflow file:** [.github/workflows/docker-publish.yml](.github/workflows/docker-publish.yml)

- **Required repository secrets (for Docker Hub push):**
   - `DOCKERHUB_USERNAME` : your Docker Hub username
   - `DOCKERHUB_TOKEN` : a Docker Hub access token (or password, but token recommended)

- **GitHub Packages (GHCR):** The workflow uses the built-in `GITHUB_TOKEN` so no additional secret is required to publish to `ghcr.io` for the same repository. Ensure you enable package publishing for your repository if your organization has additional policies.

- **How tags are mapped:**
   - Pushing a tag `v1.2.3` will publish `:v1.2.3` and update `:latest` on both registries (if Docker Hub credentials are provided).
   - Pushing to `main` without a tag publishes `:latest`.

Example: Pulling the image from GHCR

```bash
docker pull ghcr.io/<OWNER>/<REPO>:latest
# or by tag
docker pull ghcr.io/<OWNER>/<REPO>:v1.2.3
```

Example: Pulling the image from Docker Hub (if you configured `DOCKERHUB_USERNAME`):

```bash
docker pull <DOCKERHUB_USERNAME>/$(basename <REPO>):latest
```

If you'd like I can also add a `make` target or a simple GitHub action input to customize registry targets.

**Testing the workflow on a branch (no push)**

- You can test builds on a branch without publishing by using the manual workflow run and setting the `push` input to `false` (this is the default). This will run the full build (multi-arch) but will not push images to Docker Hub.

Steps to test on a branch:

1. Push your branch to GitHub, e.g. `feature/test-workflow`:

```bash
git checkout -b feature/test-workflow
git push -u origin feature/test-workflow
```

2. On GitHub, go to the `Actions` tab → select `Build and Publish Docker Hub image` → `Run workflow`.

3. In the `Run workflow` form, choose the branch `feature/test-workflow` and set the input `push` to `false`.

4. Click `Run workflow`. The job will build the image but skip login/push steps. Check logs for build output.

When you're ready to publish from that branch, re-run the workflow with `push=true` or create a tag and push the tag to trigger a publish.
