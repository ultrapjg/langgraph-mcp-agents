FROM python:3.12-slim

# Set ARG for platform detection
ARG TARGETPLATFORM
ARG BUILDPLATFORM
RUN echo "Building on $BUILDPLATFORM for $TARGETPLATFORM"

WORKDIR /app

# Install Node.js with architecture detection
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    ca-certificates \
    && if [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
       ARCH="arm64"; \
       echo "Installing Node.js for ARM64"; \
    else \
       ARCH="amd64"; \
       echo "Installing Node.js for AMD64"; \
    fi \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Verify Node.js and npm installations
RUN node --version && npm --version

# Platform-specific compiler options
ENV NODE_OPTIONS=${TARGETPLATFORM/linux\/arm64/--max_old_space_size=2048}

# Install Python dependencies with platform-specific optimizations
RUN pip install --no-cache-dir --upgrade pip && \
    if [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
        echo "Installing dependencies with ARM optimizations"; \
        pip install --no-cache-dir --extra-index-url https://www.piwheels.org/simple \
            langchain-anthropic>=0.3.10 \
            langchain-community>=0.3.20 \
            langchain-mcp-adapters>=0.0.7 \
            langchain-openai>=0.3.11 \
            langgraph>=0.3.21 \
            mcp>=1.6.0 \
            python-dotenv>=1.1.0 \
            streamlit>=1.44.1 \
            nest-asyncio>=1.5.8 \
            asyncio>=3.4.3; \
    else \
        echo "Installing dependencies with AMD optimizations"; \
        pip install --no-cache-dir \
            langchain-anthropic>=0.3.10 \
            langchain-community>=0.3.20 \
            langchain-mcp-adapters>=0.0.7 \
            langchain-openai>=0.3.11 \
            langgraph>=0.3.21 \
            mcp>=1.6.0 \
            python-dotenv>=1.1.0 \
            streamlit>=1.44.1 \
            nest-asyncio>=1.5.8 \
            asyncio>=3.4.3; \
    fi

# Copy required application files
COPY app_KOR.py mcp_server_local.py ./
COPY utils.py ./

# Create an empty .env file if needed
RUN touch .env

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PATH="/usr/local/bin:/usr/bin:${PATH}"

# Expose Streamlit port
EXPOSE 8585

# Set entry point
ENTRYPOINT ["streamlit", "run", "app_KOR.py", "--server.address=0.0.0.0", "--server.port=8585"]
