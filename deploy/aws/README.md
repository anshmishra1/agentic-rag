# AWS interview-demo deployment

This package runs AgenticRAG on one stoppable EC2 instance using the existing
locked `cpu-runtime` Docker target. API and Streamlit reuse one image;
PostgreSQL uses the official PostgreSQL image. Pinecone, Groq, and OpenRouter
remain managed external services.

## Resulting boundary

```text
Browser :8501
    |
Streamlit container
    |
private Docker network
    +-- FastAPI :8000 (not published on the host)
    +-- PostgreSQL :5432 (not published on the host)

External HTTPS: Pinecone -> Groq -> OpenRouter fallback
Persistent EBS data: PostgreSQL volume, V3 assets, application run logs
```

The application image is built directly on EC2 for the first deployment. This
avoids keeping a multi-gigabyte image in ECR while the project is used only for
short interview demonstrations. Routine starts use `--no-build`.

## 1. Secure the AWS account

1. Create or sign in to the AWS account and confirm the account registration.
2. Enable MFA on the root user. Do not create root access keys.
3. Create an administrative identity in IAM Identity Center and use it for
   routine work instead of the root user.
4. In **Billing and Cost Management -> Budgets**, create:
   - a **Zero spend budget** for Free Tier overage;
   - a small monthly cost budget, such as USD 5, with email alerts.
5. Choose one deployment region. `ap-south-1` is the documented default; use
   the same region for EC2, the IAM policy ARN, and Parameter Store.

An AWS account still requires a payment method. Free Tier or promotional
credits reduce eligible charges but do not make every resource free.

## 2. Create the EC2 runtime role

In **IAM -> Roles**, create `AgenticRagEc2Role`:

1. Trusted entity: **AWS service -> EC2**.
2. Attach `AmazonSSMManagedInstanceCore` so Session Manager can reach the host
   without opening SSH.
3. Add an inline policy based on
   [`iam-parameter-policy.template.json`](iam-parameter-policy.template.json).
   Replace `<REGION>` and `<ACCOUNT_ID>` first.
4. Create an instance profile for the role if the console does not create one
   automatically.

The inline policy grants read access only to the application's parameter path.
If SecureString values use a customer-managed KMS key, also grant
`kms:Decrypt` for that one key. The default Parameter Store key does not require
adding a broad KMS permission here.

## 3. Create runtime parameters

Open **Systems Manager -> Parameter Store** in the deployment region. Create
the following values under `/agentic-rag/prod/`:

| Name | Type | Value |
|---|---|---|
| `GROQ_API_KEY` | SecureString | Existing Groq key |
| `OPENROUTER_API_KEY` | SecureString | Existing OpenRouter key |
| `PINECONE_API_KEY` | SecureString | Existing Pinecone key |
| `POSTGRES_PASSWORD` | SecureString | New URL-safe random password |
| `PROVIDER_ORDER` | String | `groq,openrouter` |
| `PINECONE_INDEX_NAME` | String | Existing V2 index name |
| `V3_INDEX_NAME` | String | Existing V3 index name |

Use the complete name, for example
`/agentic-rag/prod/GROQ_API_KEY`. Never paste these values into Git, EC2 user
data, Compose YAML, or shell history. Additional settings from `.env.example`
may be added to the same path when an AWS-specific override is needed.

Generate the PostgreSQL password locally without printing an existing secret:

```powershell
$bytes = New-Object byte[] 24
[Security.Cryptography.RandomNumberGenerator]::Fill($bytes)
[Convert]::ToHexString($bytes).ToLowerInvariant()
```

Copy the generated result directly into the SecureString value and then clear
the PowerShell variable with `Remove-Variable bytes`.

Treat `POSTGRES_PASSWORD` as stable after the first database volume is created.
Changing only its Parameter Store value does not rotate the password inside an
existing PostgreSQL database; password rotation requires an explicit database
change followed by the Parameter Store update.

## 4. Run the read-only account preflight

Before launching EC2, open AWS CloudShell in the selected region. CloudShell is
already authenticated with the current console identity; do not configure or
paste access keys. Clone the repository after this deployment package has been
merged, then run:

```bash
git clone https://github.com/anshmishra1/agentic-rag.git
cd agentic-rag
AWS_REGION=ap-south-1 \
SSM_PARAMETER_PATH=/agentic-rag/prod \
EC2_ROLE_NAME=AgenticRagEc2Role \
bash deploy/aws/preflight.sh
```

The preflight reads metadata only. It does not decrypt Parameter Store values,
create resources, start services, or build images. It verifies:

- the authenticated account and enabled region;
- the EC2 trust relationship, Session Manager policy, instance profile, and
  exact Parameter Store path permission;
- all seven parameter names and their String/SecureString types;
- the presence of both recommended AWS budgets and a notification rule;
- existing EC2, EBS, snapshot, Elastic IP, NAT Gateway, RDS, load balancer, and
  private ECR resources that may already incur charges.

A clean new account should finish with zero failures and zero warnings. Existing
resources are warnings because they require a human cost review. The script
cannot guarantee a zero bill: billing data and budget alerts are delayed, and
resources created after the preflight can still incur charges.

## 5. Launch the EC2 instance

Recommended first deployment:

- AMI: Ubuntu Server 24.04 LTS, x86_64.
- Instance: `t3.medium` (4 GiB) minimum; `t3.large` (8 GiB) is safer while
  ingesting layout-heavy PDFs.
- Storage: 30 GiB encrypted gp3 root volume. Increase only if the CPU image,
  build cache, uploaded assets, and logs approach the limit.
- IAM instance profile: `AgenticRagEc2Role`.
- Public IP: enabled for the temporary demo; do not allocate an Elastic IP.
- Security group inbound: TCP 8501 from **My IP** only. Do not expose 8000,
  5432, or SSH. Add another `/32` only when a remote reviewer needs access.
- Security group outbound: HTTPS/DNS access so the host can install packages
  and the API can contact Pinecone and configured LLM providers.

The public IP normally changes after an instance stop/start. PostgreSQL, V3
assets, and logs stay on the EBS root volume.

## 6. Install and deploy

Connect through **EC2 -> Connect -> Session Manager**. Then run:

```bash
cd /opt
sudo git clone https://github.com/anshmishra1/agentic-rag.git
cd /opt/agentic-rag
sudo bash deploy/aws/bootstrap.sh
sudo env AWS_REGION=ap-south-1 SSM_PARAMETER_PATH=/agentic-rag/prod \
  bash deploy/aws/manage.sh deploy
```

The first `deploy` builds the CPU image and may take substantial time. Later
starts reuse that image:

```bash
cd /opt/agentic-rag
sudo env AWS_REGION=ap-south-1 SSM_PARAMETER_PATH=/agentic-rag/prod \
  bash deploy/aws/manage.sh start
```

Open `http://<EC2-public-IP>:8501` only after both services report healthy.

## 7. Operations

Run these from `/opt/agentic-rag`, retaining the same region and parameter path:

```bash
# Status
sudo env AWS_REGION=ap-south-1 SSM_PARAMETER_PATH=/agentic-rag/prod \
  bash deploy/aws/manage.sh status

# Follow logs
sudo env AWS_REGION=ap-south-1 SSM_PARAMETER_PATH=/agentic-rag/prod \
  bash deploy/aws/manage.sh logs

# Endpoint checks from inside the private containers
sudo env AWS_REGION=ap-south-1 SSM_PARAMETER_PATH=/agentic-rag/prod \
  bash deploy/aws/manage.sh health

# Stop containers without deleting data
sudo env AWS_REGION=ap-south-1 SSM_PARAMETER_PATH=/agentic-rag/prod \
  bash deploy/aws/manage.sh stop
```

Stopping containers does not stop EC2 billing. After the demo, stop the EC2
instance in the AWS console. EBS and public IPv4 pricing are separate from
instance compute; delete the instance and its EBS volume only when the stored
documents and database are no longer needed.

## 8. Updating deployed source

After a merged application change:

```bash
cd /opt/agentic-rag
sudo git pull --ff-only origin main
sudo env AWS_REGION=ap-south-1 SSM_PARAMETER_PATH=/agentic-rag/prod \
  bash deploy/aws/manage.sh build
sudo env AWS_REGION=ap-south-1 SSM_PARAMETER_PATH=/agentic-rag/prod \
  bash deploy/aws/manage.sh start
```

Configuration-only Parameter Store changes require `start`, which recreates
services only when Compose detects a changed configuration. Source or locked
dependency changes require `build` once.

## Known deployment limits

- Streamlit has no application-level login yet. Keep inbound 8501 restricted
  to known IP addresses and stop the instance after each demo.
- `t3.medium` is a capacity floor, not a performance guarantee. CPU embedding,
  Ettin reranking, PyMuPDF Layout, and RapidOCR are slower than the local CUDA
  runtime.
- The repository must choose and document a compliant AGPL or commercial path
  for PyMuPDF/PyMuPDF4LLM before treating this as an unrestricted public
  production service.
- This first package intentionally excludes ECR, ECS, RDS, ALB, Route 53,
  CloudFront, and NAT Gateway. They can be introduced after the single-host
  deployment is measured.
