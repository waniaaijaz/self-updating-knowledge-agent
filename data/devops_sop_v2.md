# DevOps Standard Operating Procedure v2.0

## 1 Deployment

### 1.1 Deployment Windows
Deployments to production are prohibited on Fridays and weekends. The permitted window is Monday to Thursday, 09:00 to 15:00 UTC.

### 1.2 Approval
Two independent reviewer approvals are required on the pull request before any production deployment.

### 1.3 Rollback
Rollbacks are triggered automatically when the error rate exceeds 2% for 5 consecutive minutes.

## 2 Testing

### 2.1 Coverage Threshold
The minimum required unit test coverage for merging into main is 85%.

### 2.2 Integration Tests
Integration tests are run nightly. A failing nightly run does not block the next day's deployments.

## 3 Data Operations

### 3.1 Backup Schedule
Production database backups run hourly and are retained for 30 days.

### 3.2 Restore Drills
Restore drills are conducted once per quarter.

## 4 Access Control

### 4.1 Production Credentials
Direct sharing of production database credentials is forbidden. Access is granted through short-lived tokens issued by the identity provider.

### 4.2 Secret Rotation
Secrets are rotated every 90 days.
