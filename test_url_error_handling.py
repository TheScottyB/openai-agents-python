#!/usr/bin/env python3
import requests
import socket
import time
import urllib.parse
from requests.exceptions import RequestException, Timeout, ConnectionError, HTTPError


def test_invalid_url_format():
    """Test handling of an invalidly formatted URL."""
    print("\n--- Testing Invalid URL Format ---")
    try:
        invalid_url = "http://invalid url with spaces"
        print(f"Testing URL: {invalid_url}")
        response = requests.get(invalid_url)
        print(f"Unexpected success with status code: {response.status_code}")
    except RequestException as e:
        print(f"✓ Successfully caught error: {type(e).__name__}")
        print(f"Error message: {str(e)}")


def test_nonexistent_domain():
    """Test handling of a URL with a non-existent domain."""
    print("\n--- Testing Non-existent Domain ---")
    try:
        nonexistent_domain = "http://this-domain-definitely-does-not-exist-123456789.com"
        print(f"Testing URL: {nonexistent_domain}")
        response = requests.get(nonexistent_domain, timeout=5)
        print(f"Unexpected success with status code: {response.status_code}")
    except ConnectionError as e:
        print(f"✓ Successfully caught connection error: {type(e).__name__}")
        print(f"Error message: {str(e)}")
    except socket.gaierror as e:
        print(f"✓ Successfully caught DNS resolution error: {type(e).__name__}")
        print(f"Error message: {str(e)}")
    except RequestException as e:
        print(f"✓ Successfully caught error: {type(e).__name__}")
        print(f"Error message: {str(e)}")


def test_nonexistent_page():
    """Test handling of a valid domain but non-existent page."""
    print("\n--- Testing Non-existent Page ---")
    try:
        nonexistent_page = "https://www.google.com/this-page-definitely-does-not-exist-123456789"
        print(f"Testing URL: {nonexistent_page}")
        response = requests.get(nonexistent_page)
        
        # This should return a 404, but still be a valid response object
        if response.status_code == 404:
            print(f"✓ Successfully received 404 status code")
            print(f"Response content length: {len(response.content)} bytes")
        else:
            print(f"Unexpected status code: {response.status_code}")
            
    except HTTPError as e:
        print(f"Caught HTTP error: {type(e).__name__}")
        print(f"Error message: {str(e)}")
    except RequestException as e:
        print(f"Caught error: {type(e).__name__}")
        print(f"Error message: {str(e)}")


def test_special_characters():
    """Test handling of a URL with special characters."""
    print("\n--- Testing URL with Special Characters ---")
    try:
        # Create a URL with various special characters that need encoding
        special_char_url = "https://www.google.com/search?q=" + urllib.parse.quote("!@#$%^&*()_+{}[]|\\:;\"'<>,.?/")
        print(f"Testing URL: {special_char_url}")
        response = requests.get(special_char_url)
        print(f"✓ Successfully handled URL with special characters. Status code: {response.status_code}")
    except RequestException as e:
        print(f"Caught error: {type(e).__name__}")
        print(f"Error message: {str(e)}")


def test_timeout():
    """Test handling of a URL that times out."""
    print("\n--- Testing URL Timeout ---")
    try:
        # This example uses a very short timeout to simulate a timeout condition
        # In real situations, you might use a known slow server
        timeout_url = "https://httpbin.org/delay/5"  # Takes 5 seconds to respond
        print(f"Testing URL: {timeout_url}")
        print("Setting a very short timeout (0.1 seconds)...")
        response = requests.get(timeout_url, timeout=0.1)
        print(f"Unexpected success with status code: {response.status_code}")
    except Timeout as e:
        print(f"✓ Successfully caught timeout error: {type(e).__name__}")
        print(f"Error message: {str(e)}")
    except RequestException as e:
        print(f"Caught other request error: {type(e).__name__}")
        print(f"Error message: {str(e)}")


def main():
    """Run all URL error handling tests."""
    print("Starting URL Error Handling Tests...")
    
    test_invalid_url_format()
    test_nonexistent_domain()
    test_nonexistent_page()
    test_special_characters()
    test_timeout()
    
    print("\nAll tests completed.")


if __name__ == "__main__":
    main()

