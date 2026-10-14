# INFRA-RESTRICTED-900: IT Infrastructure, Root Bastion & Secret Management

## 1. Production Bastion SSH Keys and JIT Access
Direct root SSH access to AWS production VPCs is strictly forbidden. IT Engineers and Cloud Ops must request Just-In-Time (JIT) access via Teleport PAM with 4-hour automatic session expiration and full session video recording. Production SSH keys rotate every 30 days automatically via HashiCorp Vault.

## 2. Kubernetes Cluster Administrative Roles
Cluster admin privileges (`system:masters`) on Amazon EKS production clusters are gated by AWS IAM Identity Center and require secondary approval from an IT Site Reliability Engineering (SRE) Lead. Ephemeral kubeconfig credentials have a maximum TTL of 60 minutes.

## 3. Disaster Recovery and Incident Break-Glass Procedures
Break-glass emergency credentials for AWS Management Console root accounts are partitioned across dual hardware security keys (YubiKey 5C) held in biometric physical safes in Austin and Dublin facilities. Unsealing requires concurrent authorization of two Designated Infrastructure Key Custodians.
