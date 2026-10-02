# DevOps Standard Operating Procedure v1.0

## 1 Deployment

### 1.1 Deployment Windows
Deployments to production are permitted on any weekday, including Fridays, between 09:00 and 17:00 UTC.

### 1.2 Approval
A single reviewer approval on the pull request is sufficient to deploy to production.

### 1.3 Rollback
Rollbacks are performed manually by the on-call engineer using the deploy CLI.

## 2 Testing

### 2.1 Coverage Threshold
The minimum required unit test coverage for merging into main is 70%.

### 2.2 Integration Tests
Integration tests are run nightly. A failing nightly run does not block the next day's deployments.

## 3 Data Operations

### 3.1 Backup Schedule
Production database backups run daily at midnight UTC and are retained for 14 days.

### 3.2 Restore Drills
Restore drills are conducted once per year.

## 4 Access Control

### 4.1 Production Credentials
Production database credentials are shared with the platform team via the team password vault.

### 4.2 Secret Rotation
Secrets are rotated every 180 days.
