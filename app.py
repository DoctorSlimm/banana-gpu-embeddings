import os
import torch
import sentry_sdk

import traceback
from dotenv import load_dotenv
from time import time
from InstructorEmbedding import INSTRUCTOR

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"


def init():
    load_dotenv()
    global model

    ##############################
    # Load the model / pipeline
    ##############################
    print('Loading model... and moving it to GPU')
    model = INSTRUCTOR('hkunlp/instructor-large')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print('Device: ', device)
    print('Model Device: ', model.device)

    ##############################
    # Initialize Sentry
    ##############################
    sentry_dsn = os.getenv('SENTRY_DSN')
    sentry_sdk.init(dsn=sentry_dsn, traces_sample_rate=0.1)


def inference(model_inputs: dict) -> dict:
    start_time = time()
    try:
        global model

        # Parse out your arguments
        ping = model_inputs.get('ping', None)
        if ping is not None:
            return {'message': 'pong'}

        device_name = model_inputs.get('device_name', None)
        if device_name is not None:
            device = torch.device(device_name)
            model.to(device)

        inputs = model_inputs.get('inputs', None)
        if inputs is None:
            return {
                'message': "No inputs provided"
            }
        num_inputs = len(inputs)
        print(f"Received {num_inputs} inputs")

        ######################################
        # Run the model
        ######################################
        t0 = time()
        result = model.encode(
            inputs,
            show_progress_bar=True,
        ).tolist()

        ######################################
        # Return the results as a dictionary
        ######################################
        return {
            'message': 'Success',
            'result': result,
            'time': time() - t0,
            'total_time': time() - start_time,
            'device': str(model.device),
        }

    except Exception as e:
        print(traceback.format_exc())
        print(e)
        return {
            'error': str(traceback.format_exc()) + str(e)
        }
