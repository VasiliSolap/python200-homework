# Warmup 08 - Cloud Computing

## Cloud Concepts

### Q1 - Economic Model
Cloud computing works on a pay-as-you-go model — you pay only for what you use,
like electricity. This is operational expenditure  instead of capital
expenditure. Instead of buying expensive servers, companies rent cloud
resources for the time they need and scale as necessary.

### Q2 - Vertical vs Horizontal Scaling
Vertical scaling means making one server more powerful by adding CPU, RAM, or
better GPU. Horizontal scaling means adding more servers to distribute the load.

- Scenario 1: Horizontal — need to handle more parallel requests, distribute
  across servers.
- Scenario 2: Vertical — need one more powerful machine with better GPU and RAM
  for model training.
- Scenario 3: Horizontal — 10,000 files can be split across machines and
  processed in parallel.

### Q3 - IaaS, PaaS, SaaS

Classification:
- Gmail → SaaS
- Azure Virtual Machines → IaaS
- Azure App Service → PaaS
- AWS S3 → IaaS
- GitHub Codespaces → PaaS
- Snowflake → SaaS

IaaS — Infrastructure as a Service: you get basic resources (servers, storage)
and configure everything else yourself. Example: Azure Virtual Machines.

PaaS — Platform as a Service: the platform is ready, you only write and deploy
code. Example: Azure App Service.

SaaS — Software as a Service: ready-made product, you just use it, nothing to
configure. Example: Gmail, Snowflake.

### Q4 - Managed Data Platforms
Managed data platforms like Databricks and Snowflake are pre-configured services
for working with data. Unlike using Azure directly where you create VMs, configure
networks and install tools yourself — everything is ready here. You gain speed and
simplicity but lose flexibility and pay more.

### Q5 - When Cloud is Not the Right Choice
Cloud is not the right choice in two cases: first — when the workload is small
and stable, and it is cheaper to keep a local server than pay for cloud monthly.
Second — when data requires strict confidentiality (medical or military data) and
cannot be stored on third-party servers.

## Azure Basics

### Q1 - Subscription vs Resource Group
Subscription is the billing account that all resources are tied to. CTD has one
subscription for all students — CTD Nonprofit Sponsorship. Resource Group is a
personal folder inside the subscription where only your resources live. Each
student has their own Resource Group (p200-2026-vasili-rg), but the Subscription
is shared across all of CTD.

### Q2 - Ephemeral vs Persistent
Ephemeral means Cloud Shell is temporary — every time you close the browser the
container is deleted and all files disappear. To make it persistent, we connected
a Storage Account (vasilictd2026sa) — a separate storage that lives independently
of the container. Now all files including SSH keys are saved between sessions.

### Q3 - SSH Keys
The private key (id_rsa) stays only with you and is never transmitted. The public
key (id_rsa.pub) is uploaded to servers you want to connect to. When connecting,
the server checks mathematically if the keys match — without transmitting a
password over the network. This is safe because even if someone sees the public
key — without the private key it is impossible to connect.

### Q4 - az account show
Output without flag:
{
  "environmentName": "AzureCloud",
  "homeTenantId": "0f040ddd-301f-4665-8677-7b21f129d605",
  "id": "4e07c58c-751e-4765-b40c-632b9ee6fe6e",
  "isDefault": true,
  "managedByTenants": [],
  "name": "CTD Nonprofit Sponsorship",
  "state": "Enabled",
  "tenantId": "0f040ddd-301f-4665-8677-7b21f129d605",
  "user": {
    "cloudShellID": true,
    "name": "live.com#vasilisolap@gmail.com",
    "type": "user"
  }
}

Without the flag the command returns data in JSON format — convenient for scripts
and programs. Adding --output table displays the same result as a readable table
with columns — convenient for quick viewing by a person.
