# Deploying HireLoop to AWS App Runner

HireLoop's default mode is local (stdio, data on your machine). This guide hosts it as a **remote MCP server** (streamable HTTP) on AWS App Runner, using Amazon Bedrock for grading and embeddings.

> **Status:** the HTTP transport, auth and `/health` were tested locally. The `Dockerfile` and the AWS steps below have **not** been run end to end (the Docker daemon was not running when this was written, and no AWS account was used). Treat them as a checked draft and expect small fixes.

## What changes when it is remote

| Concern | Local (stdio) | Remote (App Runner) |
|---|---|---|
| Transport | stdio | streamable HTTP at `/mcp` |
| Auth | none needed | **bearer token required** (the server refuses to start on a non-loopback host without `HIRELOOP_AUTH_TOKEN`) |
| LLM grading / embeddings | Groq + local fastembed | Bedrock (Converse + Titan), Groq fallback |
| Data | `~/.hireloop/hireloop.db` | container disk: **ephemeral** (see limits) |

## Limits to know before demoing

- **Single user.** There is one profile row and one shared database, so everyone with the token shares the same data. Fine for a demo; not multi-tenant.
- **Data is lost on redeploy or restart.** App Runner has no persistent volume, so SQLite lives on ephemeral disk. The planned fix is a DynamoDB implementation behind `store.py`, which was designed as a thin swappable layer.
- **Privacy story changes.** "Your data stays on your machine" holds for the local mode only. In remote mode the data sits in your AWS account.
- Keep `min instances = max instances = 1` so all requests hit the same database.

## 1. Enable Bedrock models

In the Bedrock console (same region as the service), request access to:
- a text model for grading, and
- **Titan Text Embeddings V2** (`amazon.titan-embed-text-v2:0`).

Set `BEDROCK_MODEL_ID` to a model ID or inference-profile ID you have access to (check the Bedrock console's model catalogue; newer models often need a regional inference-profile ID). If a call fails, HireLoop logs it and falls back to Groq, and `submit_answer` reports which model graded (`graded_by`).

## 2. Build and push the image

```powershell
$REGION = "us-east-1"
$ACCOUNT = (aws sts get-caller-identity --query Account --output text)
$REPO = "$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/hireloop"

aws ecr create-repository --repository-name hireloop --region $REGION
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin "$ACCOUNT.dkr.ecr.$REGION.amazonaws.com"
docker build -t hireloop .
docker tag hireloop:latest "${REPO}:latest"
docker push "${REPO}:latest"
```

Test the image locally first:

```powershell
docker run --rm -p 8000:8000 -e HIRELOOP_AUTH_TOKEN=test-token -e USE_BEDROCK=false -e GROQ_API_KEY=... hireloop
curl http://localhost:8000/health
```

## 3. Secrets

Create secrets in AWS Secrets Manager (do not put them in the image or the repo):

```powershell
python -c "import secrets; print(secrets.token_urlsafe(32))"   # use as the auth token
aws secretsmanager create-secret --name hireloop/auth-token --secret-string "<token>"
aws secretsmanager create-secret --name hireloop/groq-key --secret-string "<groq key>"
aws secretsmanager create-secret --name hireloop/elevenlabs-key --secret-string "<elevenlabs key>"
```

## 4. IAM roles

**Instance role** (what the running app may do), trust principal `tasks.apprunner.amazonaws.com`:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": ["bedrock:InvokeModel", "bedrock:Converse"],
    "Resource": "*"
  }]
}
```

Tighten `Resource` to the specific model and inference-profile ARNs once it works. No AWS keys are needed in the environment: boto3 picks up the instance role automatically.

**Access role** for pulling from ECR: the managed policy `AWSAppRunnerServicePolicyForECRAccess`, trust principal `build.apprunner.amazonaws.com`.

To read the secrets above at runtime, the instance role also needs `secretsmanager:GetSecretValue` on those secrets.

## 5. Create the service

In the App Runner console (or `aws apprunner create-service`):

| Setting | Value |
|---|---|
| Source | ECR image `hireloop:latest` |
| Port | `8000` |
| Health check | HTTP, path `/health` (open without a token) |
| Instances | min 1, max 1 |
| Instance role | the role from step 4 |

Environment variables:

```
USE_BEDROCK=true
AWS_REGION=us-east-1
BEDROCK_MODEL_ID=<your model or inference profile id>
BEDROCK_EMBED_MODEL_ID=amazon.titan-embed-text-v2:0
GROQ_MODEL=openai/gpt-oss-120b
```

Runtime secrets (from Secrets Manager): `HIRELOOP_AUTH_TOKEN`, `GROQ_API_KEY`, and optionally `ELEVENLABS_API_KEY` plus `ELEVENLABS_VOICE_ID`.

Note: audio from `start_mock_interview` is written to the container's disk and the returned path only exists on the server, so voice interviews are really a local-mode feature until audio moves to S3 (see Roadmap).

## 6. Connect a client

The endpoint is `https://<service-id>.<region>.awsapprunner.com/mcp`. Every request needs `Authorization: Bearer <token>`.

Quick check:

```powershell
curl https://<service-url>/health
curl -i https://<service-url>/mcp        # expect 401 without the token
```

Claude Desktop reads stdio servers from its config, so reach a remote one through the `mcp-remote` proxy (verify the flags against its current docs):

```json
{
  "mcpServers": {
    "hireloop-remote": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "https://<service-url>/mcp",
               "--header", "Authorization: Bearer ${HIRELOOP_TOKEN}"],
      "env": { "HIRELOOP_TOKEN": "<token>" }
    }
  }
}
```

## Verifying Bedrock is really used

Run a mock interview and check `graded_by` in the `submit_answer` result: `bedrock:<model id>` means Bedrock graded it; `groq:<model>` means it fell back (the server log says why, for example "no AWS credentials found" or an access-denied error).
