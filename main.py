import streamlit as st
from langchain_experimental.agents import create_csv_agent
from langchain_huggingface import HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI

def main():
    st.header("Chat with Spread sheet")
    
    upload_file = st.file_uploader('upload the csv file' , type= 'csv')
    if upload_file is not None:
        llm = HuggingFaceEndpoint(
            repo_id='meta-llama/Llama-3.2-3B-Instruct',
            temperature=0.3,
            max_new_tokens= 1024,
            stop=["\n"],
            huggingfacehub_api_token='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'
        )
        agent = create_csv_agent(llm , upload_file , verbose = True , allow_dangerous_code = True , handle_parsing_errors=True)
        
        question = st.text_input('Enter the question here :')
        if question is not None and question!= '':
            with st.spinner(text = 'In Progress...'):
                st.write(agent.run(question))
                
if __name__ == '__main__':
    main()
    
            
        
        