import os
import torch
import logging
import sentry_sdk

import traceback
from dotenv import load_dotenv
from time import time
from InstructorEmbedding import INSTRUCTOR

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"


def init():
    global model
    try:
        start_time = time()
        load_dotenv()

        ##############################
        # Load the model / pipeline
        ##############################
        logging.info('Loading model... and moving it to GPU')
        model = INSTRUCTOR('hkunlp/instructor-large')

        ##############################
        # Move model to GPU
        ##############################
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        logging.info('Device: ', device)
        logging.info('Model Device: ', model.device)

        ##############################
        # Initialize Sentry
        ##############################
        logging.info('Initializing Sentry')
        sentry_dsn = os.getenv('SENTRY_DSN')
        sentry_sdk.init(dsn=sentry_dsn, traces_sample_rate=0.1)

        logging.info('Initialization complete in {} seconds'.format(time() - start_time))

    except Exception as e:
        logging.error(traceback.format_exc())
        logging.error(e)
        raise e


def inference(model_inputs: dict) -> dict:
    start_time = time()
    try:
        global model

        ######################################
        # Parse arguments
        ######################################
        ping = model_inputs.get('ping', None)
        if ping is not None:
            logging.info('Ping received')
            return {
                'message': 'pong',
                'total_time': time() - start_time,
            }

        inputs = model_inputs.get('inputs', None)
        if inputs is None:
            return {
                'message': "No inputs provided"
            }

        device_name = model_inputs.get('device_name', None)
        if device_name is not None:
            logging.info(f"Changing device to {device_name}")
            device = torch.device(device_name)
            model.to(device)

        model_inputs.pop('inputs', None)
        num_inputs = len(inputs)
        logging.info(f"Received Arguments: {model_inputs}")
        logging.info(f"Received {num_inputs} inputs")

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
        logging.error(traceback.format_exc())
        logging.error(e)
        return {
            'error': str(traceback.format_exc()) + str(e)
        }
