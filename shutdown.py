import os
import requests
import sys
import time
import argparse
from datetime import datetime, timedelta

def parse_arguments():
    parser = argparse.ArgumentParser(description="Schedule pod shutdown after specified hours")
    parser.add_argument("hours", type=float, help="Number of hours to wait before shutting down")
    parser.add_argument("--dry-run", action="store_true", help="Print shutdown time without actually scheduling")
    return parser.parse_args()

def shutdown_pod():
    # Get API Key and Pod ID from environment variables
    api_key = os.environ.get("RUNPOD_API_KEY")
    pod_id = os.environ.get("RUNPOD_POD_ID")

    if not api_key:
        print("Error: RUNPOD_API_KEY environment variable not found.")
        sys.exit(1)

    if not pod_id:
        print("Error: Could not retrieve RUNPOD_POD_ID.")
        sys.exit(1)

    # First, let's check if the pod exists and get its status
    pod_url = f"https://api.runpod.io/v2/pod/{pod_id}"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    print(f"Checking pod status for: {pod_id}...")
    pod_response = requests.get(pod_url, headers=headers)
    
    if pod_response.status_code == 404:
        print(f"Pod {pod_id} not found. It may have been already stopped/deleted.")
        print("Available pods:")
        list_pods(api_key)
        return
    elif pod_response.status_code != 200:
        print(f"Failed to get pod info. Status code: {pod_response.status_code}")
        print(f"Response: {pod_response.text}")
        return
    
    pod_info = pod_response.json()
    print(f"Pod status: {pod_info.get('desiredStatus', 'unknown')}")
    
    # Only try to stop if pod is running
    if pod_info.get('desiredStatus') == 'STOPPED':
        print("Pod is already stopped.")
        return

    # Construct the API endpoint URL for stopping the pod
    stop_url = f"https://api.runpod.io/v2/pod/{pod_id}/stop"

    # Make the POST request to stop the pod
    print(f"Sending stop request for pod: {pod_id}...")
    response = requests.post(stop_url, headers=headers)

    # Check the response
    if response.status_code == 200:
        print("Successfully initiated pod stop.")
    else:
        print(f"Failed to stop pod. Status code: {response.status_code}")
        print(f"Response: {response.text}")

def list_pods(api_key):
    """List all available pods for debugging"""
    try:
        headers = {
            "Authorization": f"Bearer {api_key}"
        }
        response = requests.get("https://api.runpod.io/v2/pod", headers=headers)
        
        if response.status_code == 200:
            pods = response.json()
            if pods:
                print("Available pods:")
                for pod in pods:
                    pod_id = pod.get('id', 'unknown')
                    status = pod.get('desiredStatus', 'unknown')
                    print(f"  - {pod_id}: {status}")
            else:
                print("No pods found.")
        else:
            print(f"Failed to list pods. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error listing pods: {e}")

def main():
    args = parse_arguments()
    
    # Calculate shutdown time
    shutdown_time = datetime.now() + timedelta(hours=args.hours)
    
    print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Scheduled shutdown time: {shutdown_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Waiting {args.hours} hours ({args.hours * 3600:.0f} seconds)...")
    
    if args.dry_run:
        print("Dry run mode - not actually scheduling shutdown")
        return
    
    # Wait for the specified time
    time.sleep(args.hours * 3600)
    
    # Execute shutdown
    print(f"Shutdown time reached. Stopping pod...")
    shutdown_pod()

if __name__ == "__main__":
    main()