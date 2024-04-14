import sys
from garmin_fit_sdk import Decoder, Stream, Profile
import pandas as pd
import os
import io

import subprocess

# Function to process the data
def process_data(data):
    print("got data in process here")
    # Perform calculations or other operations here
    return decode_fit_buffer(data)

def decode_fit_buffer(buffer_data):
    base_path = "./Data"
    file_name = "decoded_fit_data"
    out_file = os.path.join(base_path, file_name + ".csv")

    # Convert buffer to bytes and create a stream
    fit_bytes = bytes(buffer_data)
    buffered_reader = io.BufferedReader(io.BytesIO(bytearray(buffer_data)))
    stream = Stream.from_buffered_reader(buffered_reader)
    assert stream.get_buffered_reader() is not None
    assert stream.peek_byte() == 0x0E
    decoder = Decoder(stream)

    record_fields = set()

    def mesg_listener(mesg_num, message):
        if mesg_num == Profile['mesg_num']['RECORD']:
            for field in message:
                record_fields.add(field)

    messages, errors = decoder.read(apply_scale_and_offset=True,
                                    convert_types_to_strings=True,
                                    enable_crc_check=True,
                                    expand_sub_fields=True,
                                    expand_components=True,
                                    merge_heart_rates=True,
                                    mesg_listener=mesg_listener)

    if len(errors) > 0:
        print(f"Something went wrong decoding the file: {errors}")

    print(record_fields)
    df = pd.DataFrame(messages['record_mesgs'])
    df.to_csv(out_file)
    # Call EffortDistributionAnalysis.py
    subprocess.run(["python", "EffortDistributionAnalysis.py"])

# Read array buffer from stdin
data = sys.stdin.buffer.read()

# Process the data
processed_data = process_data(data)

# Send the processed data to stdout
# sys.stdout.buffer.write("Done processing data")
# send the confirmation message to the client
sys.stdout.buffer.write(b"Done processing data from the server\n")
sys.stdout.flush()
