import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from clients.proxy_client import ProxyClient, VLLM_CONFIGS

def verify_configs():
    print("Verifying vLLM configurations...")
    
    # Verify specific models
    for model, config in VLLM_CONFIGS.items():
        print(f"Testing model: {model}")
        try:
            client = ProxyClient(model=model)
            
            # Check base_url
            # OpenAI client stores base_url as a URL object, convert to string
            actual_url = str(client.client.base_url)
            expected_url = config['base_url']
            
            # OpenAI client might append a trailing slash if not present, or handle it differently
            # Let's normalize by stripping trailing slashes for comparison
            if actual_url.rstrip('/') != expected_url.rstrip('/'):
                print(f"❌ Mismatch for {model}:")
                print(f"   Expected: {expected_url}")
                print(f"   Actual:   {actual_url}")
                return False
            
            print(f"✅ {model} passed. URL: {actual_url}")
            
        except Exception as e:
            print(f"❌ Error initializing {model}: {e}")
            return False
            
    # Verify fallback
    print("\nTesting fallback model: qwen3vl-other")
    try:
        # Set env var to ensure deterministic fallback
        os.environ["VLLM_BASE_URL"] = "http://fallback-url:8000/v1"
        client = ProxyClient(model="qwen3vl-other")
        
        actual_url = str(client.client.base_url)
        expected_url = "http://fallback-url:8000/v1"
        
        if actual_url.rstrip('/') != expected_url.rstrip('/'):
            print(f"❌ Fallback mismatch:")
            print(f"   Expected: {expected_url}")
            print(f"   Actual:   {actual_url}")
            return False
            
        print(f"✅ Fallback passed. URL: {actual_url}")
        
    except Exception as e:
        print(f"❌ Error initializing fallback: {e}")
        return False
        
    return True

if __name__ == "__main__":
    if verify_configs():
        print("\nAll verifications passed!")
        sys.exit(0)
    else:
        print("\nVerification failed!")
        sys.exit(1)
