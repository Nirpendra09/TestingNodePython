import sys

# Function to process the data
def process_data(data):
    print("got data in process here")
    # Perform calculations or other operations here
    return data[::-1]  # Example: Reverse the array buffer

# Read array buffer from stdin
data = sys.stdin.buffer.read()

# Process the data
processed_data = process_data(data)

# Send the processed data to stdout
sys.stdout.buffer.write(processed_data)
sys.stdout.flush()
