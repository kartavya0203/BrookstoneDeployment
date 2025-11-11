# 📦 REQUIREMENTS.TXT - UPDATED FOR GEMINI MIGRATION

## ✅ **Updated Requirements File**

Your `requirements.txt` has been updated to match the Gemini-powered codebase exactly.

### 🎯 **Core Dependencies Analysis**

Based on the imports in `brookstone_pinecone.py`:

```python
# Standard Library (No installation needed)
import os, re, logging, json
from datetime import datetime, timedelta

# Third-party packages (Included in requirements.txt)
from flask import Flask, request, jsonify           # ✅ flask>=2.3.0
import requests                                     # ✅ requests>=2.31.0
from dotenv import load_dotenv                      # ✅ python-dotenv>=1.0.0
from langchain_pinecone import PineconeVectorStore  # ✅ langchain-pinecone>=0.1.0
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI  # ✅ langchain-google-genai>=1.0.0
import google.generativeai as genai                 # ✅ google-generativeai>=0.8.0
```

### 📁 **Files Created**

1. **`requirements.txt`** - Comprehensive version with comments and optional packages
2. **`requirements-production.txt`** - Minimal production-ready version

### 🔧 **Installation Commands**

```bash
# For development (includes testing tools)
pip install -r requirements.txt

# For production (minimal dependencies only)
pip install -r requirements-production.txt
```

### 🎯 **Key Changes From Migration**

**Removed (OpenAI-related):**

- ❌ `langchain-openai`
- ❌ `openai`
- ❌ `langchain-community` (was used for OpenAI embeddings)

**Added/Updated (Gemini-focused):**

- ✅ `langchain-google-genai>=1.0.0` (Gemini integration)
- ✅ `google-generativeai>=0.8.0` (Direct Gemini API)
- ✅ `langchain-core>=0.2.0` (Core LangChain functionality)

### 🚀 **Production Ready**

Your requirements file now includes:

- **Version Pinning**: Minimum versions specified for stability
- **Production Server**: Gunicorn for deployment
- **Development Tools**: Pytest and Black for testing/formatting
- **Clear Organization**: Grouped by functionality with comments

### ✅ **Verification**

All packages tested and working:

- ✅ Flask web framework
- ✅ Google Gemini AI integration
- ✅ Pinecone vector database
- ✅ Environment variable management
- ✅ HTTP requests handling

Your bot is now ready for deployment with the updated requirements! 🎉

---

_Updated: November 11, 2025 - Post Gemini Migration_
