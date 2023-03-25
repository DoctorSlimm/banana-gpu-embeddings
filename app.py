import torch
import traceback
from time import time
from InstructorEmbedding import INSTRUCTOR

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"


def init():
    global model

    model = INSTRUCTOR('hkunlp/instructor-large')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print('Device: ', device)
    print('Model Device: ', model.device)


def inference(model_inputs: dict) -> dict:
    try:
        global model

        # Parse out your arguments
        inputs = model_inputs.get('inputs', None)
        if inputs is None:
            return {'message': "No inputs provided"}
        num_inputs = len(inputs)
        print(f"Received {num_inputs} inputs")

        # Run the model
        t0 = time()
        result = model.encode(
            inputs,
            show_progress_bar=True,
        ).tolist()

        # Return the results as a dictionary
        return {
            'message': 'Success',
            'result': result,
            'time': time() - t0
        }

    except Exception as e:
        print(traceback.format_exc())
        print(e)
        return {
            'error': str(traceback.format_exc()) + str(e)
        }
