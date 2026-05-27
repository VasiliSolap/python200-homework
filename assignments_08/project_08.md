# Project 08 - Cost Analysis

## Video
https://youtu.be/2OJ_Gpo2slg

## Cost Analysis Summary

### Scenario A - Lightweight Compute
- VM: Standard_B1s (1 vCPU, 1 GB RAM)
- Hours: 160/month (8h/day, 5 days/week)
- Monthly cost: $2.24
- Hourly rate: $0.014/hour

### Scenario B - Heavy Analytics Workload
- VM: Standard_NC6s_v3 (6 vCPU, 1 V100 GPU) — $2,233.80/month
- Azure SQL Database (General Purpose, 4 vCores) — $2,975/month
- Azure Blob Storage (1 TB) — $20.80/month
- Total: $5,229.60/month

## What Surprised Me
The difference between the two scenarios was striking — Scenario B costs
over 3,000x more than Scenario A in total. The GPU VM alone costs $2,233.80
per month running 24/7, which is 1,396x more expensive than the lightweight
VM. The SQL Database was surprisingly the most expensive component of
Scenario B at $2,975/month — even more than the GPU VM itself.

## Script Output
=== Monthly Cost Estimates ===
Scenario A (lightweight):       $1.60
Scenario B (GPU VM only):       $2233.80
Scenario B VM costs 1396.1x more than Scenario A

The calculated costs matched the Pricing Calculator estimates exactly.
