#!/bin/bash

# GCP Deployment Script for Learning Platform
# Run this in Google Cloud Shell

set -e  # Exit on error

echo "🚀 Starting GCP Deployment..."
echo ""

# Configuration
PROJECT_ID="capstone-1218-eagly"
REGION="asia-south1"
SERVICE_NAME="learning-platform"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "📋 Configuration:"
echo "  Project ID: ${PROJECT_ID}"
echo "  Region: ${REGION}"
echo "  Service: ${SERVICE_NAME}"
echo ""

# Set project
echo "🔧 Setting GCP project..."
gcloud config set project ${PROJECT_ID}

# Enable required APIs (if not already enabled)
echo "🔌 Enabling required APIs..."
gcloud services enable \
  cloudbuild.googleapis.com \
  run.googleapis.com \
  secretmanager.googleapis.com \
  storage-api.googleapis.com

# Build the container
echo "🏗️  Building container image..."
gcloud builds submit --tag ${IMAGE_NAME}

# Deploy to Cloud Run
echo "🚀 Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
  --image ${IMAGE_NAME} \
  --platform managed \
  --region ${REGION} \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 300 \
  --max-instances 10 \
  --min-instances 0 \
  --set-env-vars="PORT=8080" \
  --set-secrets="GROQ_API_KEY=GROQ_API_KEY:latest,DEEPSEEK_API_KEY=DEEPSEEK_API_KEY:latest,GEMINI_API_KEY=GEMINI_API_KEY:latest,SESSION_SECRET=SESSION_SECRET:latest,SMTP_EMAIL=SMTP_EMAIL:latest,SMTP_PASSWORD=SMTP_PASSWORD:latest,GOOGLE_CLIENT_ID=GOOGLE_CLIENT_ID:latest,GOOGLE_CLIENT_SECRET=GOOGLE_CLIENT_SECRET:latest,DATABASE_URL=DATABASE_URL:latest,GCP_BUCKET_NAME=GCP_BUCKET_NAME:latest,GOOGLE_APPLICATION_CREDENTIALS_JSON=GOOGLE_APPLICATION_CREDENTIALS_JSON:latest"

echo ""
echo "✅ Deployment complete!"
echo ""
echo "🌐 Your app URL:"
gcloud run services describe ${SERVICE_NAME} --region ${REGION} --format='value(status.url)'
echo ""
echo "📊 To view logs:"
echo "  gcloud run logs read ${SERVICE_NAME} --region ${REGION}"
echo ""
echo "💰 To check costs:"
echo "  https://console.cloud.google.com/billing"
echo ""
