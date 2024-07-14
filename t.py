import subprocess
import datetime
import os

def take_screenshot():
    # Create a timestamp for the screenshot filename
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    screenshot_filename = f"screenshot_{timestamp}.png"

    try:
        # Take a screenshot using adb shell command
        result = subprocess.run(['adb', 'exec-out', 'screencap', '-p'], capture_output=True, text=False)

        if result.returncode == 0:
            # Save the screenshot to a file
            with open(screenshot_filename, 'wb') as f:
                f.write(result.stdout)
            print(f"Screenshot saved to {screenshot_filename}")
        else:
            print(f"Failed to take screenshot. Error: {result.stderr.decode('utf-8')}")
    except FileNotFoundError:
        print("adb is not installed or not found in your PATH.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Example usage
take_screenshot()
