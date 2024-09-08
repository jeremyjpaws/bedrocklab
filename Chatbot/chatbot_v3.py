import json
import boto3
import os
import requests
from requests_aws4auth import AWS4Auth

region = os.environ.get('AWS_REGION')
bedrock = boto3.client(service_name='bedrock-runtime')
aoss_host = os.environ.get('aossHost')

def lambda_handler(event, context):
    if (event['httpMethod'] == 'GET'):
        output = load_html()
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'text/html'},
            'body': output
        }
    elif (event['httpMethod'] == 'POST'):
        prompt = event['body']
        comm, query = check_prompt_command(prompt)
        if comm == 'search':
            output = search(query, limit=5)
        elif comm == 'rag':
            # retrieve the context
            info = search(query, limit=5)
            # augment the prompt with the context
            prompt = 'Use the context below to answer the question:\n\n=== Context ===\n{0}\n\n=== Question ===\n{1}'.format(info, query)
            output = chat(prompt)
        else:
            output = chat(prompt)
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'text/html'},
            'body': output
        }
    else:
         return {
            'statusCode': 200,
            'headers': {'Content-Type': 'text/html'},
            'body': "OK"
        }

def check_prompt_command(prompt):
    comm = 'chat'
    query = None
    # Check the last line of the prompt to see if it is a search request.
    lines = prompt.splitlines()
    last_line = lines[-1]
    # Check if the last line starts with "Human: "
    if last_line.startswith('Human: '):
        last_line = last_line[7:].strip()
        # Check if the human prompt starts with "//search "
        if last_line.startswith('//search '):
            query = last_line[9:].strip()
            if query != None:
                comm = 'search'
        # Check if the human prompt starts with "//rag "
        if last_line.startswith('//rag '):
            query = last_line[5:].strip()
            if query != None:
                comm = 'rag'
    return comm, query

def load_html():
    html = ''
    with open('index.html', 'r') as file:
        html = file.read()
    return html

def chat(prompt):
    modelId = 'ai21.j2-ultra'
    accept = 'application/json'
    contentType = 'application/json'
    body=json.dumps({'prompt': prompt, 'maxTokens': 250, 'stopSequences': ['Human:']})
    response = bedrock.invoke_model(body=body, modelId=modelId, accept=accept,contentType=contentType)
    response_body = json.loads(response.get('body').read())
    completions = response_body['completions']
    output = ''
    for part in completions:
        output += part['data']['text']
    return output

def search(query, limit=1):
    # get embedding
    embedding = get_embedding(query)
    # prepare for OpenSearch Serverless
    service = 'aoss'
    credentials = boto3.Session().get_credentials()
    awsauth = AWS4Auth(
        credentials.access_key, 
        credentials.secret_key, 
        region, 
        service, 
        session_token=credentials.token
    )
    # search
    index = 'demo-index'
    datatype = '_search'
    url = aoss_host + '/' + index + '/' + datatype
    headers = {'Content-Type': 'application/json'}
    document = {
        'size': limit,
        'query': {
            'knn': {
                'embedding': {
                    'vector': embedding,
                    'k': limit
                }
            }
        }
    }
    # response
    response = requests.get(url, auth=awsauth, json=document, headers=headers)
    response.raise_for_status()
    data = response.json()
    output = ''
    for item in data['hits']['hits']:
        output += item['_source']['content'] + '\n'
    return output.strip()

def get_embedding(text):
    modelId = 'amazon.titan-embed-text-v1'
    accept = 'application/json'
    contentType = 'application/json'
    input = {
            'inputText': text
        }
    body=json.dumps(input)
    response = bedrock.invoke_model(
        body=body, modelId=modelId, accept=accept,contentType=contentType)
    response_body = json.loads(response.get('body').read())
    embedding = response_body['embedding']
    return embedding
