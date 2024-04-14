from garmin_fit_sdk import Decoder, Stream, Profile
import  pandas as pd

import os

def get_base_path_and_filename(file_path):
  """Extracts the base path and filename without extension from a given file path.

  Args:
    file_path (str): The full path to the file.

  Returns:
    tuple: A tuple containing the base path (str) and filename without extension (str).
  """
  base_path = os.path.dirname(file_path)
  filename_with_ext = os.path.basename(file_path)
  filename_without_ext = os.path.splitext(filename_with_ext)[0]
  return base_path, filename_without_ext




file_path = "./Data/2508278192_ACTIVITY.fit"
base_path, file_name = get_base_path_and_filename(file_path)

out_file = base_path + "/" + file_name + ".csv"
stream = Stream.from_file(file_path)
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