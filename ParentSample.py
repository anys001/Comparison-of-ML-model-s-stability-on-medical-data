import os
import subprocess
import json
import time
import pickle
import argparse

MARK = b"<<<Done!>>>"


# Creating parser
parser = argparse.ArgumentParser(description='Parameters manager.')

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# giving arguments
parser.add_argument('--data', type=str, required=True, help='Way to raw data')
# parser.add_argument('--transformer', type=str, required=True, help='Way to script of data prepare')
parser.add_argument('--model_name', type=str, required=True, help='Model name')
parser.add_argument('--optimization', type=str2bool, default=False, help='Optimize in every step')
# Example usage --data data3.csv --model_name Regression --optimization N
# Arguments and its usage
args = parser.parse_args()

print('\nRun parameters:\n', args)

TRANSFORMERS_MAPPING = {
    'data.csv': {
        "trans": "transformer_01.py",
        "res": "ready_data_01.pkl",
    },

    'data3.csv': {
        "trans": "transformer_03.py",
        "res": "ready_data_03.pkl",
    },
}

file_path_trans = TRANSFORMERS_MAPPING[args.data]['res']

# File existance check
if not os.path.exists(file_path_trans):
    # Run child
    child = subprocess.Popen(['python', f"./{TRANSFORMERS_MAPPING[args.data]['trans']}"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    print(f"Started <{child.pid}> as a child!")

    # ============ Preparing data ==============
    try:
        # Child process
        print(f"\nSending to transformer<{child.pid}>")
        child.stdin.write(args.data + "\n")
        child.stdin.writelines(TRANSFORMERS_MAPPING[args.data]['res'] + "\n")
        child.stdin.flush()
        print(f"\t Sent! Waiting a response...")

        # Loop to read from child process
        while True:
            # line = child.stdout.readline()
            line = child.stdout.buffer.readline()
            if not line: break

            if line.startswith(MARK):
                print(f"Got message from child: {line}")
                break
            else:
                print(f"Flood:::\n{line}")
    except KeyboardInterrupt:
        pass
    finally:
        # Closing process
        child.stdout.close()  # closing stdout
        child.wait()
        child.terminate()

# ============ Run experiment ==============
child = subprocess.Popen([
        'python', r'./run_functions.py', 
        "--data",
        file_path_trans,
        "--model_name",
        args.model_name,
        "--opt",
        "Y" if args.optimization else "N",
        "--cpu_load",
        "0.9",
        "--seed",
        "42",
    ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
print(f"Started <{child.pid}> as a child!")

try:
    print("Experiment is started...")
    while True:
        line = child.stdout.buffer.readline()
        if not line:
            break

        # check message from child
        if line.startswith(MARK):
            print(f"Got message from child: {line}")
            break
        else:
            print(f"Flood:::\n{line}")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    print("Closing streams...")
    child.stdout.close()
    error = child.stderr.readline()
    child.stderr.close()
    print("Closing the child...")
    child.wait()
    print("Exiting...")
    exit()
