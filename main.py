import argparse
import os
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="Simple Docker test script")
    parser.add_argument("--message", type=str, default="Hello from inside the container!")
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Print message and save to a file
    print(args.message)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = os.path.join(args.output_dir, f"message_{timestamp}.txt")

    with open(output_path, "w") as f:
        f.write(args.message + "\n")

    print(f"✅ Message saved to {output_path}")

if __name__ == "__main__":
    main()
