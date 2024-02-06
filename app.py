import streamlit as st
import pandas as pd
from ast import literal_eval
import chromadb
from chromadb.utils import embedding_functions
import openai
from pathlib import Path
import base64
import uuid
import json
from openai import AzureOpenAI
from chromadb import Documents, EmbeddingFunction, Embeddings


def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

def img_to_html(img_path):
    img_html = "<img src='data:image/png;base64,{}' height = '80' width = '180'>".format(
      img_to_bytes(img_path)
    )
    return img_html

st.set_page_config(layout="wide")
padding_top = 0

st.markdown(f"""

<style>

    .appview-container .main .block-container{{

        padding-top: 0;
        margin: 0;
        height: 250%;
        overflow: auto;

    }}

    .st-bt {{
    background-color: #e8eae9;
    }}

</style>""",

            unsafe_allow_html=True,

            )
st.text("")
st.markdown("""
    <style>

        .background {
        background-color: rgb(241, 237, 238);
        padding: 10px;
        margin-top: 1%;
        border: 1px solid #ccc;
        box-shadow: 4px 4px 5px rgba(0, 0, 0, 0.3);
        overflow: hidden;
        }

        .title_heading {
        color: #000000;
        font-size: 22px;
        font-weight: bold;
        font-family: "Open Sans", sans-serif;
        }
        .title {
        margin-top: 20px;
        display: flex;
        }
        .button-inline {
        color: green;
        background-color: rgb(241, 237, 238);
        padding: 10px 20px;
        font-size: 11px;
        font-weight: bold;
        border: 1px solid white;
        margin-left: auto;
        margin-right: 10px;
        height: 20px;
        margin-top: 1px;
        line-height: 0.3;
        border-radius: 5px;
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3);
        }

        """

    f"""</style>
    <div class="background">
        <p class="title">
        <img src='data:image/png;base64,{img_to_bytes("Picture1.png")}' height = '35' width = '140'>
        <span class ="title_heading"> | TMNA </span>
        <button class="button-inline" type="button">Logout</button>
    </p>
    </div>
        """,

            unsafe_allow_html=True,

            )


##### uncomment later
##Create a text container with a black background
st.markdown("""
    <style>

        .background_black {
        background-color: #000000;
        padding-top: 0px;
        border: 1px solid #ccc;
        box-shadow: 4px 4px 5px rgba(0, 0, 0, 0.3);
        margin-top: 1%;
        margin-bottom: -3%;
        position: relative;
        }

        .paragraph_heading {
        color: rgb(134, 188, 37);
        font-size: 18px;
        font-weight: bold;
        font-family: "Open Sans", sans-serif;
        }

        .paragraph_body {
        color: #ffffff;
        font-size: 14px;
        font-weight: bold;
        font-family: "Open Sans", sans-serif;
        }
        .paragraph {
        margin-left: 20px;
        margin-top: 10px;
        }
        .image{
          position: absolute;
          top: 8;
          right: 0;
          margin-left: 10px;
        }
        .help-image{

          position: absolute;
          right: 35px; /* Adjust the value to shift the image to the right by the desired amount */
          top: 12px; 

        }

        .help-icon{

          position: absolute;
          right: -46px; /* Adjust the value to shift the image to the right by the desired amount */
          top: 47px; 

        }

    </style>"""
    f"""<div class="background_black">
    <p class="paragraph">
        <span class ="paragraph_heading">Business Rule Extraction</span><br>
        <span class ="paragraph_body">A Generative AI powered QnA application</span>
        <img src='data:image/png;base64,{img_to_bytes("help logo.png")}' height = '40' width = '40' class = "help-image"> <br>
        <img src='data:image/png;base64,{img_to_bytes("Help.png")}' height = '30' width = '120' class = "help-icon">
    </p>
    </div>
        """,

            unsafe_allow_html=True,

            )

def retreive_relevant_docs(chromdb_collection,prompt_embeddings):
    title_query_result = chromdb_collection.query(
        query_embeddings=prompt_embeddings,
        n_results=10,
        include=['metadatas','distances', 'documents']
    )
#     print(title_query_result)
    result = pd.DataFrame({
                'id':title_query_result['ids'][0], 
                'score':title_query_result['distances'][0],
                'docs':title_query_result['documents'][0],
                'metadata': title_query_result['metadatas'][0]
                })
    return result


def get_embeddings(input_text):
    client = AzureOpenAI(
    api_key = "3a3b78172a564e5187901b0b2a268100",
    api_version = "2023-05-15",
    azure_endpoint = "https://usammcomponentfactoryopenaiservicetmna.openai.azure.com/"
    )
    
    response = client.embeddings.create(
        input = input_text,
        model= "TextEmbeddingADA002"
    )
    
    return response.data[0].embedding

# Function to style the DataFrame
def highlight_max(s):
    is_max = s == s.max()
    return ['background-color: yellow' if v else '' for v in is_max]


def main():
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
        
    df = pd.read_csv('refined_embeddings.csv')
    df['embeddings'] = df['embeddings'].apply(literal_eval)
    
    # metadf = df[['Annotation_Type','Category','State','Rule_Sequence_Number']]
    # data = metadf.to_dict(orient='records')
    result_df = df[['program_name','Source Code','Rule_Sequence_Number','Annotation_Description','Functional_Description', 'embeddings','id']]
    result_df['id'] = result_df['id'].apply(str)
    
    

    chroma_client = chromadb.Client()
    chromdb_collection = chroma_client.get_or_create_collection(name='stories')
    
    metadf = df[['program_name','Source Code','Rule_Sequence_Number','Annotation_Description']]
    data = metadf.to_dict(orient='records')
    
    chromdb_collection.add(
    ids=result_df.id.tolist(),
    embeddings=result_df.embeddings.tolist(),
    metadatas=data,
    documents=result_df.Functional_Description.tolist()
    )
    
    col1, col2, col3 = st.columns([5,3,3])

    with col1:
        prompt1 = st.button("List all business rules related to production planning", on_click=clear_text)
    with col2:
        prompt2 = st.button("List all business rules related to SSN validation", on_click=clear_text)
    with col3:
        prompt3 = st.button("List all business rules related to production year", on_click=clear_text)

    
    # user_query = [st.text_input('Write your prompt')]
    try:
        # Get text input from the user
        user_query = None
        user_query_input = str(st.text_input("Enter some text:", key = 'text'))

        # Check if there is no input
        if user_query_input:
            user_query = user_query_input
            # print('inside 1111111111111')
        elif prompt1: 
            user_query = "List all business rules related to production planning"
            # print('inside 222222222222222')
            
        elif prompt2:
            user_query = "List all business rules related to SSN validation"
            # print('inside 33333333333333333')
        
        elif prompt3:
            user_query = "List all business rules related to production periods"
            # print('inside 444444444444444')
            
        if user_query:
            print('*************user query:*************** ', user_query)
            title_query_result = retreive_relevant_docs(chromdb_collection, get_embeddings(user_query))
            print('*************relevant documents**************')
            print(title_query_result)
            # st.write(title_query_result['documents'])
            context = create_context(title_query_result)
            # print(context)

            # generate RAG responses by passing top relevant documents
            system_prompt = """
            You are human friendly conversational assistant that writes natural, relevant and meaningful responses to the user questions based on the context information provided by the user. 
            
            """

            message_prompt = f"""
            User Question:
            ```
            {user_query}
            ```

            Context:
            ```
            {context}
            ```

            Human friendly Answer:
            First summarize business rule in short and then explain business rules that are functionally belongs to the user question without splitting the business rules. Explanation should be more detailed, concrete and natural in nature. Explain these business rules individually.
            
            You should never mention program name, rule sequence number, rule number and document number and any other information from the provided context while explaining business rules.
            
            Do not explain business rules that do not belong to the user question.

            At the end, you must explicitly display actual program name and rule sequence number to the user as part of citation from the provided context. Do not repeat the same information again. Do not forget to mention this citation information.

            Make sure you generate detailed and meaningful responses and keep it more natural & conversational.
            """

            response = get_response(message_prompt, system_prompt, temperature = 0.7)


            st.write(response)

            
            # similar_rules = json.loads(response)

            # sub_dfs = []
            # for info in similar_rules['results']:

            #     df = result_df[result_df['id'] == info['id']][['program_name', 'Source Code', 'Rule_Sequence_Number', 'Functional_Description'
            #                                                     ]]
            #     df['similarity score'] = info['score']

            #     sub_dfs.append(df)

            # preview_df = pd.concat(sub_dfs, ignore_index=True)
            # preview_df = preview_df.sort_values(by=['similarity score'], ascending=False)
            # preview_df = preview_df.reset_index(drop=True)

            # st.markdown("<p style='color: green; font-size: 25px; text-align: center;'>Similarity Results</p>", unsafe_allow_html=True)
            # # Convert the DataFrame to HTML without index
            # html_table = preview_df.to_html(index=False, escape=False)

            # # Display the HTML table in Streamlit
            # st.markdown(html_table, unsafe_allow_html=True)


    except Exception as e:
        print(e)

def clear_text():
    st.session_state["text"] = ""

def get_response(prompt, system, temperature = 1.0):
    client = AzureOpenAI(
    api_key = "3a3b78172a564e5187901b0b2a268100",  
    api_version = "2023-05-15",
    azure_endpoint = "https://usammcomponentfactoryopenaiservicetmna.openai.azure.com/"
    )

    response = client.chat.completions.create(
        model="TurboCodeGPT35Turbo", # model = "deployment_name".
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        max_tokens=4096
    )

    #print(response)
    # print(response.model_dump_json(indent=2))
    return response.choices[0].message.content

def create_context(df):
    context = ''
    # print(df['documents'])
    for i, row in df[['metadata', 'docs']].iterrows():
        
        context += f"Document {i+1} : \nProgram Name : {row['metadata']['program_name']}\nRule Sequence no. : {row['metadata']['Rule_Sequence_Number']}\nBusiness Rule : {row['docs']}\n\n"
        
#     print('context: ', context)
    return context

if __name__ == "__main__":
    main()
