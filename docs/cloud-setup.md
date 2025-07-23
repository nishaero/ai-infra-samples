# Cloud Setup Guide

This document provides step-by-step instructions for setting up the necessary cloud infrastructure and configurations for the Climate Temperature Prediction project.

## Prerequisites

Before starting, ensure you have:

- AWS Account with appropriate permissions
- AWS CLI installed and configured
- Terraform >= 1.0 installed
- Docker installed
- Git installed
- Python 3.9+ installed

## AWS Account Setup

### 1. Create AWS Account

If you don't have an AWS account:
1. Go to [AWS Console](https://aws.amazon.com/console/)
2. Click "Create an AWS Account"
3. Follow the registration process
4. Verify your email and phone number
5. Add a payment method (free tier eligible resources will not incur charges)

### 2. Create IAM User for Development

Instead of using root credentials, create a dedicated IAM user:

1. **Log in to AWS Console**
2. **Navigate to IAM service**
3. **Create a new user:**
   ```bash
   # Via AWS CLI (after configuring with root/admin credentials)
   aws iam create-user --user-name climate-prediction-dev
   ```

4. **Attach necessary policies:**
   ```bash
   # Attach required policies for the project
   aws iam attach-user-policy \
     --user-name climate-prediction-dev \
     --policy-arn arn:aws:iam::aws:policy/AmazonEC2FullAccess
   
   aws iam attach-user-policy \
     --user-name climate-prediction-dev \
     --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
   
   aws iam attach-user-policy \
     --user-name climate-prediction-dev \
     --policy-arn arn:aws:iam::aws:policy/AmazonECS_FullAccess
   
   aws iam attach-user-policy \
     --user-name climate-prediction-dev \
     --policy-arn arn:aws:iam::aws:policy/AmazonRDSFullAccess
   
   aws iam attach-user-policy \
     --user-name climate-prediction-dev \
     --policy-arn arn:aws:iam::aws:policy/IAMFullAccess
   ```

5. **Create access keys:**
   ```bash
   aws iam create-access-key --user-name climate-prediction-dev
   ```

   **⚠️ Important:** Save the Access Key ID and Secret Access Key securely. You won't be able to retrieve the secret key again.

### 3. Configure AWS CLI

Configure your local AWS CLI with the new credentials:

```bash
aws configure
# Enter your Access Key ID
# Enter your Secret Access Key
# Enter your preferred region (e.g., us-west-2)
# Enter default output format (json)
```

Verify the configuration:
```bash
aws sts get-caller-identity
```

## NASA Earthdata Setup

### 1. Create NASA Earthdata Account

1. Go to [NASA Earthdata Login](https://urs.earthdata.nasa.gov/)
2. Click "Register for a profile"
3. Fill out the registration form
4. Verify your email address
5. Complete your profile information

### 2. Configure NASA Earthdata Credentials

Set up environment variables for NASA Earthdata access:

```bash
# Add to your shell profile (.bashrc, .zshrc, etc.)
export EARTHDATA_USERNAME="your_nasa_username"
export EARTHDATA_PASSWORD="your_nasa_password"
```

Or create a `.env` file in the project root:
```bash
# .env file
EARTHDATA_USERNAME=your_nasa_username
EARTHDATA_PASSWORD=your_nasa_password
```

## Project Setup

### 1. Clone the Repository

```bash
git clone https://github.com/nishaero/ai-infra-samples.git
cd ai-infra-samples
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
make install
```

### 3. Configure Environment Variables

Create a `.env` file with the necessary configuration:

```bash
# .env file
AWS_REGION=us-west-2
AWS_ACCESS_KEY_ID=your_access_key_id
AWS_SECRET_ACCESS_KEY=your_secret_access_key

EARTHDATA_USERNAME=your_nasa_username
EARTHDATA_PASSWORD=your_nasa_password

# Database configuration (will be created by Terraform)
DB_USERNAME=airflow
DB_PASSWORD=changeme123

# MLflow configuration
MLFLOW_TRACKING_URI=http://localhost:5000

# Project configuration
PROJECT_NAME=climate-prediction
ENVIRONMENT=dev
```

## Infrastructure Deployment

### 1. Initialize Terraform

```bash
cd infrastructure/terraform
terraform init
```

### 2. Plan Infrastructure

```bash
# Review what will be created
terraform plan -var="environment=dev"
```

### 3. Deploy Infrastructure

```bash
# Apply the infrastructure
terraform apply -var="environment=dev"
```

This will create:
- VPC with public and private subnets
- Security groups
- RDS PostgreSQL database
- S3 buckets for data, models, and artifacts
- ECR repositories
- ECS cluster
- Application Load Balancer
- IAM roles and policies

### 4. Get Infrastructure Outputs

```bash
# Get important outputs
terraform output alb_dns_name
terraform output s3_data_bucket
terraform output s3_models_bucket
terraform output ecr_api_repository_url
```

## Docker Setup

### 1. Build Docker Images

```bash
# Build API image
docker build -f docker/Dockerfile.api -t climate-prediction-api .

# Build training image
docker build -f docker/Dockerfile.training -t climate-prediction-training .
```

### 2. Push Images to ECR

```bash
# Get ECR login
aws ecr get-login-password --region us-west-2 | \
  docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-west-2.amazonaws.com

# Tag and push API image
docker tag climate-prediction-api:latest <account-id>.dkr.ecr.us-west-2.amazonaws.com/climate-prediction/api:latest
docker push <account-id>.dkr.ecr.us-west-2.amazonaws.com/climate-prediction/api:latest

# Tag and push training image
docker tag climate-prediction-training:latest <account-id>.dkr.ecr.us-west-2.amazonaws.com/climate-prediction/training:latest
docker push <account-id>.dkr.ecr.us-west-2.amazonaws.com/climate-prediction/training:latest
```

## Local Development Setup

### 1. Start Local Services

```bash
# Start all services locally
make dev
```

This starts:
- MLflow tracking server (http://localhost:5000)
- Airflow webserver (http://localhost:8080)
- Climate Prediction API (http://localhost:8000)
- Prometheus (http://localhost:9090)
- Grafana (http://localhost:3000)

### 2. Access Services

- **API Documentation:** http://localhost:8000/docs
- **MLflow UI:** http://localhost:5000
- **Airflow UI:** http://localhost:8080 (admin/admin)
- **Grafana Dashboard:** http://localhost:3000 (admin/admin)
- **Prometheus:** http://localhost:9090

## CI/CD Setup (GitHub Actions)

### 1. Configure GitHub Secrets

In your GitHub repository, add the following secrets:

1. Go to Repository Settings → Secrets and variables → Actions
2. Add the following secrets:

```
AWS_ACCESS_KEY_ID=your_access_key_id
AWS_SECRET_ACCESS_KEY=your_secret_access_key
EARTHDATA_USERNAME=your_nasa_username
EARTHDATA_PASSWORD=your_nasa_password
```

### 2. Trigger CI/CD Pipeline

The CI/CD pipeline will automatically run on:
- Push to main branch
- Pull requests to main branch
- Manual trigger

## Monitoring and Alerting

### 1. CloudWatch Setup

The Terraform configuration automatically sets up:
- CloudWatch log groups for ECS tasks
- CloudWatch alarms for CPU and memory usage
- SNS topics for alerting

### 2. Configure Alert Notifications

```bash
# Subscribe to SNS topic for alerts
aws sns subscribe \
  --topic-arn arn:aws:sns:us-west-2:123456789012:climate-prediction-alerts \
  --protocol email \
  --notification-endpoint your-email@example.com
```

## Cost Optimization

### Free Tier Resources Used

The project is designed to use AWS Free Tier resources where possible:

- **EC2:** Fargate tasks under free tier limits
- **RDS:** db.t3.micro instance (free tier eligible)
- **S3:** 5GB of standard storage
- **Data Transfer:** 15GB outbound per month
- **CloudWatch:** 10 custom metrics and 5GB log ingestion

### Estimated Monthly Costs

For light usage (development/testing):
- **RDS (db.t3.micro):** $0 (free tier) / $13.50 after free tier
- **S3 Storage:** $0-5 depending on data volume
- **ECS Fargate:** $0-20 depending on task runtime
- **Data Transfer:** $0 (within free tier limits)
- **Load Balancer:** ~$16.20/month
- **NAT Gateway:** ~$32.40/month

**Estimated Total:** $50-75/month after free tier expires

### Cost Reduction Tips

1. **Use Spot Instances:** For training workloads
2. **Schedule Resources:** Stop non-production resources when not needed
3. **Optimize Storage:** Use S3 Intelligent Tiering
4. **Monitor Usage:** Set up billing alerts

## Troubleshooting

### Common Issues

1. **Terraform Apply Fails:**
   ```bash
   # Check AWS credentials
   aws sts get-caller-identity
   
   # Check region configuration
   aws configure get region
   ```

2. **Docker Push Fails:**
   ```bash
   # Refresh ECR login
   aws ecr get-login-password --region us-west-2 | \
     docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-west-2.amazonaws.com
   ```

3. **API Health Check Fails:**
   ```bash
   # Check ECS service status
   aws ecs describe-services \
     --cluster climate-prediction-cluster \
     --services climate-prediction-api
   ```

4. **Training Job Fails:**
   ```bash
   # Check ECS task logs
   aws logs describe-log-streams \
     --log-group-name /ecs/climate-prediction
   ```

### Getting Help

- **AWS Support:** Use AWS Support Center for infrastructure issues
- **GitHub Issues:** Report bugs or feature requests
- **Documentation:** Check the project README and architecture docs

## Next Steps

After completing the setup:

1. **Run the Training Pipeline:** Trigger a manual training run
2. **Test the API:** Make prediction requests to validate functionality
3. **Set Up Monitoring:** Configure dashboards and alerts
4. **Scale as Needed:** Adjust ECS task counts and resource allocations

## Security Best Practices

1. **Use IAM Roles:** Avoid hardcoding credentials
2. **Enable MFA:** For AWS root and admin accounts
3. **Regular Key Rotation:** Rotate access keys quarterly
4. **VPC Security:** Use private subnets for sensitive resources
5. **Data Encryption:** Enable encryption at rest and in transit
6. **Monitor Access:** Use CloudTrail for audit logging