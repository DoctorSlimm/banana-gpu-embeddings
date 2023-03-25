# Loading model weights

# In this file, we define download_model()
#   which will be called in the Dockerfile

# Alternatively,
#   can fetch the model weights in the Dockerfile (S3, git-lfs, etc.)
#   and save them into cache directories (e.g. ~/.cache/torch/transformers/)


import torch
from time import time
from InstructorEmbedding import INSTRUCTOR


sample_inputs = [
    ['Represent the Wikipedia question for retrieving supporting documents: ','where is the food stored in a yam plant'],
    ['Represent the Wikipedia document for retrieval: ','Capitalism has been dominant in the Western world since the end of feudalism, but most feel[who?] that the term "mixed economies" more precisely describes most contemporary economies, due to their containing both private-owned and state-owned enterprises. In capitalism, prices determine the demand-supply scale. For example, higher demand for certain goods and services lead to higher prices and lower demand for certain goods lead to lower prices.'],
    ['Represent the Wikipedia document for retrieval: ',"The disparate impact theory is especially controversial under the Fair Housing Act because the Act regulates many activities relating to housing, insurance, and mortgage loansâ€”and some scholars have argued that the theory's use under the Fair Housing Act, combined with extensions of the Community Reinvestment Act, contributed to rise of sub-prime lending and the crash of the U.S. housing market and ensuing global economic recession"],
    ['Represent the Wikipedia document for retrieval: ','Disparate impact in United States labor law refers to practices in employment, housing, and other areas that adversely affect one group of people of a protected characteristic more than another, even though rules applied by employers or landlords are formally neutral. Although the protected classes vary by statute, most federal civil rights laws protect based on race, color, religion, national origin, and sex as protected traits, and some laws include disability status and other traits as well.'],
    ['Represent the Wikipedia question for retrieving supporting documents: ','where is the food stored in a yam plant'],
    ['Represent the Wikipedia document for retrieval: ','Capitalism has been dominant in the Western world since the end of feudalism, but most feel[who?] that the term "mixed economies" more precisely describes most contemporary economies, due to their containing both private-owned and state-owned enterprises. In capitalism, prices determine the demand-supply scale. For example, higher demand for certain goods and services lead to higher prices and lower demand for certain goods lead to lower prices.'],
    ['Represent the Wikipedia document for retrieval: ',"The disparate impact theory is especially controversial under the Fair Housing Act because the Act regulates many activities relating to housing, insurance, and mortgage loansâ€”and some scholars have argued that the theory's use under the Fair Housing Act, combined with extensions of the Community Reinvestment Act, contributed to rise of sub-prime lending and the crash of the U.S. housing market and ensuing global economic recession"],
    ['Represent the Wikipedia document for retrieval: ','Disparate impact in United States labor law refers to practices in employment, housing, and other areas that adversely affect one group of people of a protected characteristic more than another, even though rules applied by employers or landlords are formally neutral. Although the protected classes vary by statute, most federal civil rights laws protect based on race, color, religion, national origin, and sex as protected traits, and some laws include disability status and other traits as well.'],
    ['Represent the Science sentence: ','Parton energy loss in QCD matter'],
    ['Represent the Financial statement: ','The Federal Reserve on Wednesday raised its benchmark interest rate.'],
    ['Represent the Science sentence: ','The Chiral Phase Transition in Dissipative Dynamics'],
    ['Represent the Financial statement: ','The funds rose less than 0.5 per cent on Friday']
]

for _ in range(3):
    sample_inputs += sample_inputs


def download_model():
    print("Downloading weights...\n")
    model = INSTRUCTOR('hkunlp/instructor-large')

    # Sample run inference on CPU
    device = torch.device("cpu")
    model.to(device)

    print('Sample run inference on CPU')
    t0 = time()
    outputs = model.encode(
        sample_inputs,
        show_progress_bar=True,
    ).tolist()
    print(f"Time elapsed: {time() - t0:.2f} seconds\n")

    # Sample run inference on GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print('Sample run inference on GPU')
    t0 = time()
    outputs = model.encode(
        sample_inputs,
        show_progress_bar=True,
    ).tolist()
    print(f"Time elapsed: {time() - t0:.2f} seconds\n")


if __name__ == "__main__":
    download_model()
