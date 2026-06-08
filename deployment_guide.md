# Streamlit Chatbot Cloud Deployment Guide

This guide describes how to deploy your Mixture of Experts (MoE) AI Agent Chatbot to the cloud for free using **Streamlit Community Cloud** or **Hugging Face Spaces**.

---

## 🚀 Option 1: Streamlit Community Cloud (Recommended)

Streamlit Community Cloud connects directly to your GitHub repository and deploys the app in one click.

### Step 1: Create a `.gitignore` file
To prevent committing your local secrets or temporary databases, create a `.gitignore` file in your project directory:

```
# Secrets and configs
.env
.streamlit/secrets.toml

# Vector memory database
vector_store.json

# Temporary uploads folder
uploaded_files/

# Python Cache
__pycache__/
*.pyc
.pytest_cache/
```

### Step 2: Push your code to GitHub
1. Create a new repository on GitHub (e.g. name it `moe-ai-agent-chatbot`).
2. Run these commands in your project terminal:
   ```bash
   git init
   git add .
   git commit -m "initial commit: web agent chatbot"
   git branch -M main
   git remote add origin https://github.com/your-username/moe-ai-agent-chatbot.git
   git push -u origin main
   ```

### Step 3: Deploy to Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
2. Click **New app**.
3. Select your repository: `your-username/moe-ai-agent-chatbot`.
4. Set the **Main file path** to: `app.py`.
5. Click on **Advanced settings...** to manage API keys securely:
   * Under **Secrets (TOML format)**, paste your API keys:
     ```toml
     GROQ_API_KEY = "your_groq_api_key_here"
     # GEMINI_API_KEY = "your_gemini_api_key_if_used_for_embeddings"
     ```
   * Click **Save**.
6. Click **Deploy!**

Your application will boot up in less than a minute and offer a public URL (e.g., `https://moe-ai-agent.streamlit.app/`) that is live on the internet!

---

## 🤗 Option 2: Hugging Face Spaces

If you prefer Hugging Face, you can host it as a Streamlit Space.

1. Create a Hugging Face Account and go to [Hugging Face Spaces](https://huggingface.co/spaces).
2. Click **Create new Space**.
3. Fill in the details:
   * **Space Name**: `moe-ai-agent-chatbot`
   * **SDK**: Select **Streamlit**.
   * **Space License**: Select `Apache 2.0` or custom.
4. Go to **Settings** of your Space, scroll to **Variables and secrets**, click **New secret**:
   * Key: `GROQ_API_KEY`
   * Value: `your_groq_api_key_here`
5. Clone the space repository locally, copy all the files from `E:\Agent Builder` into the Space repository (ensuring `.gitignore` is set up), commit, and push!
