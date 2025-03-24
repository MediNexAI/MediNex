# MediNex AI API Documentation

This directory contains documentation for the MediNex AI API endpoints.

## API Endpoints

### Health Check
- **Endpoint**: `/health`
- **Method**: GET
- **Description**: Check the health status of the API
- **Response**: JSON object with status information

### Medical Query
- **Endpoint**: `/api/query`
- **Method**: POST
- **Description**: Query the medical knowledge base and get an AI-generated response
- **Request Body**:
  ```json
  {
    "query": "What are the symptoms of diabetes?",
    "category": "endocrinology",
    "include_sources": true
  }
  ```
- **Response**:
  ```json
  {
    "answer": "Diabetes is characterized by several symptoms...",
    "sources": [
      {
        "title": "Diabetes Symptoms",
        "source": "Medical Encyclopedia",
        "url": "https://api.medinex.life/v1/knowledge/123"
      }
    ]
  }
  ```

### Image Analysis
- **Endpoint**: `/api/analyze-image`
- **Method**: POST
- **Description**: Analyze a medical image and get an AI-generated interpretation
- **Request Body**:
  ```json
  {
    "image": "base64_encoded_image_data",
    "model_type": "xray",
    "prompt": "Describe what you see in this chest X-ray."
  }
  ```
- **Response**: JSON object with analysis, model information, and predictions

## Authentication

All API endpoints require authentication using an API key. The API key should be included in the request headers:

```
Authorization: Bearer YOUR_API_KEY
```

## Rate Limiting

The API has rate limiting to prevent abuse. Current limits are:
- 100 requests per minute per API key
- 5,000 requests per day per API key

## Error Codes

- `400`: Bad Request - Invalid parameters
- `401`: Unauthorized - Missing or invalid API key
- `403`: Forbidden - Not authorized to access this resource
- `404`: Not Found - Resource not found
- `422`: Validation Error - Invalid request data
- `429`: Too Many Requests - Rate limit exceeded
- `500`: Internal Server Error - Server-side error

## API Versioning

The API uses versioning to ensure backward compatibility. The current version is v1, which is included in the URL path:

```
https://api.medinex.example.com/v1/query
```

For detailed examples and implementation guidelines, see the individual endpoint documentation files in this directory.

### Example Request: Analyze an image

**Endpoint**: `POST /imaging/analyze`

**Request**:
```bash
curl -X POST "https://api.medinex.life/v1/imaging/analyze" \
  -H "Content-Type: multipart/form-data" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "image=@chest_xray.jpg" \
  -F "modality=xray" \
  -F "analysis_type=diagnostic" \
  -F "clinical_context=Patient has persistent cough for 3 weeks"
``` 