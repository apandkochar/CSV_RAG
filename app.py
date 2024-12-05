import streamlit as st
from langchain_experimental.agents import create_csv_agent
from langchain_huggingface import HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI
import tempfile

def main():
    st.header("Chat with Spreadsheet")
    
    # Upload CSV file
    upload_file = st.file_uploader('Upload the CSV file', type='csv')
    if upload_file is not None:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            temp_file.write(upload_file.read())
            temp_file_path = temp_file.name

        # Initialize LLM
        try:
            # llm = ChatGoogleGenerativeAI(
            #     model = ''
            #     temperature=0.3,
            #     max_tokens=2048,
            #     api_key='',  # Replace with your actual API key
            # )
            llm = ChatGoogleGenerativeAI( model = 'gemini-1.5-flash' , temperature= 0.7 , api_key='')
        except Exception as e:
            st.error(f"Failed to initialize LLM: {e}")
            return

        # Create CSV agent
        try:
            agent = create_csv_agent(
                llm=llm, 
                path=temp_file_path, 
                verbose=True, 
                allow_dangerous_code=True
            )
        except Exception as e:
            st.error(f"Failed to create agent: {e}")
            return

        # Input question
        question = st.text_input('Enter the question here:')
        if question.strip():  # Check for non-empty input
            with st.spinner(text='In Progress...'):
                try:
                    response = agent.run(question)
                    st.write(response)
                except Exception as e:
                    st.error(f"Failed to process the question: {e}")

if __name__ == '__main__':
    main()
