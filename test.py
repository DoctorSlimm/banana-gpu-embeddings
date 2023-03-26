# This file is used to verify your http server acts as expected
# Run it with `python3 test.py``

import requests

sample_inputs = [
    ['Represent the Wikipedia question for retrieving supporting documents: ', 'where is the food stored in a yam plant'],
    ['Represent the Wikipedia document for retrieval: ', 'Capitalism has been dominant in the Western world since the end of feudalism, but most feel[who?] that the term "mixed economies" more precisely describes most contemporary economies, due to their containing both private-owned and state-owned enterprises. In capitalism, prices determine the demand-supply scale. For example, higher demand for certain goods and services lead to higher prices and lower demand for certain goods lead to lower prices.'],
    ['Represent the Wikipedia document for retrieval: ', "The disparate impact theory is especially controversial under the Fair Housing Act because the Act regulates many activities relating to housing, insurance, and mortgage loansâ€”and some scholars have argued that the theory's use under the Fair Housing Act, combined with extensions of the Community Reinvestment Act, contributed to rise of sub-prime lending and the crash of the U.S. housing market and ensuing global economic recession"],
    ['Represent the Wikipedia document for retrieval: ', 'Disparate impact in United States labor law refers to practices in employment, housing, and other areas that adversely affect one group of people of a protected characteristic more than another, even though rules applied by employers or landlords are formally neutral. Although the protected classes vary by statute, most federal civil rights laws protect based on race, color, religion, national origin, and sex as protected traits, and some laws include disability status and other traits as well.'],
    ['Represent the Wikipedia question for retrieving supporting documents: ', 'where is the food stored in a yam plant'],
    ['Represent the Wikipedia document for retrieval: ', 'Capitalism has been dominant in the Western world since the end of feudalism, but most feel[who?] that the term "mixed economies" more precisely describes most contemporary economies, due to their containing both private-owned and state-owned enterprises. In capitalism, prices determine the demand-supply scale. For example, higher demand for certain goods and services lead to higher prices and lower demand for certain goods lead to lower prices.'],
    ['Represent the Wikipedia document for retrieval: ', "The disparate impact theory is especially controversial under the Fair Housing Act because the Act regulates many activities relating to housing, insurance, and mortgage loansâ€”and some scholars have argued that the theory's use under the Fair Housing Act, combined with extensions of the Community Reinvestment Act, contributed to rise of sub-prime lending and the crash of the U.S. housing market and ensuing global economic recession"],
    ['Represent the Wikipedia document for retrieval: ', 'Disparate impact in United States labor law refers to practices in employment, housing, and other areas that adversely affect one group of people of a protected characteristic more than another, even though rules applied by employers or landlords are formally neutral. Although the protected classes vary by statute, most federal civil rights laws protect based on race, color, religion, national origin, and sex as protected traits, and some laws include disability status and other traits as well.'],
    ['Represent the Science sentence: ', 'Parton energy loss in QCD matter'],
    ['Represent the Financial statement: ', 'The Federal Reserve on Wednesday raised its benchmark interest rate.'],
    ['Represent the Science sentence: ', 'The Chiral Phase Transition in Dissipative Dynamics'],
    ['Represent the Financial statement: ', 'The funds rose less than 0.5 per cent on Friday']
]

# model_inputs = {'inputs': sample_inputs}
#
# res = requests.post('http://localhost:8000/', json=model_inputs)
#
# print(res.json())


if __name__ == '__main__':
    import os
    import banana_dev as banana
    from time import time
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv('BANANA_API_KEY')
    print('ApiKey: ', api_key)
    model_key = os.getenv('BANANA_MODEL_ID')
    print('ModelKey: ', model_key)

    for _ in range(6):
        sample_inputs += sample_inputs
    print(len(sample_inputs))

    model_inputs = {
        'inputs': sample_inputs
    }

    # GPU API
    t0 = time()
    out = banana.run(api_key, model_key, model_inputs)
    print('GPU API: ({:.2f}) / total'.format(time() - t0))
    print('GPU API: ({}) / compute'.format(out['modelOutputs'][0]['time']))
    print('Examples: {}\n'.format(len(sample_inputs)))

    # CPU (local)
    t0 = time()
    res = requests.post('http://localhost:8000/', json=model_inputs)
    print('CPU (local): ({:.2f})'.format(time() - t0))
    print('Examples: {}\n'.format(len(sample_inputs)))

