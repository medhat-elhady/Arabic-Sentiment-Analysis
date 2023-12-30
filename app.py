
import streamlit as st
import time
from google.cloud import aiplatform
import numpy as np
st.title('arabic sentiment Classifier')


# [START aiplatform_predict_custom_trained_model_sample]
from typing import Dict, List, Union

from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value


def predict_custom_trained_model_sample(
    project: str,
    endpoint_id: str,
    instances: Union[Dict, List[Dict]],
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",
):
    """
    `instances` can be either single instance of type dict or a list
    of instances.
    """
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    # The format of each instance should conform to the deployed model's prediction input schema.
    instances = instances if isinstance(instances, list) else [instances]
    instances = [
        json_format.ParseDict(instance_dict, Value()) for instance_dict in instances
    ]
    parameters_dict = {}
    parameters = json_format.ParseDict(parameters_dict, Value())
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    
    # The predictions are a google.protobuf.Value representation of the model's predictions.
    predictions = response.predictions
    return predictions


# [END aiplatform_predict_custom_trained_model_sample]
st.markdown("Welcome")


def main():
    feedback = st.text_input('Feedback')
    class_btn = st.button("Classify") 
        
    if class_btn:
        if feedback is None:
            st.write("Invalid command, please type a feedback")
        else:
            #with st.spinner('Model working....'):
                
            predictions = predict(feedback)
            st.success('Classified')
            st.write(predictions)
            
            if predictions=='Postive':
                st.image('./emotions/happy.jpg')
            else:
                st.image('./emotions/sad.png')


def predict(text):

    res = predict_custom_trained_model_sample(
    project="1033908600341",
    endpoint_id="1064408619547623424",
    location="us-central1",
    instances={ "text": text}
)
    if np.round(res[0][0]) == 1.0:
        return 'Postive'
    else:
        return 'Negative'
    
    

if __name__ == "__main__":
    main()
