from langchain_experimental.agents import create_csv_agent
from langchain_community.llms import OpenAI
import streamlit as st
import os
import tempfile

# Set the OpenAI API key to an empty string (You should replace this with your actual API key)
os.environ["OPENAI_API_KEY"] = ""

def main():
    # Configure Streamlit page
    st.set_page_config(page_title="Ask your CSV")
    st.header("Ask your CSV")

    # Allow the user to upload a CSV file
    file = st.file_uploader("upload file", type="csv")

    if file is not None:
        # Create a temporary file to store the uploaded CSV data
        with tempfile.NamedTemporaryFile(mode='w+', suffix=".csv", delete=False) as f:
            # Convert bytes to a string before writing to the file
            data_str = file.getvalue().decode('utf-8')
            f.write(data_str)
            f.flush()

            # Create an instance of the OpenAI language model with temperature set to 0
            llm = OpenAI(temperature=0)

            # Ask the user to input a question
            user_input = st.text_input("Question here:")

            # Create a CSV agent using the OpenAI language model and the temporary file
            agent = create_csv_agent(llm, f.name, verbose=True)

            if user_input:
                # Run the agent on the user's question and get the response
                response = agent.run(user_input)
                st.write(response)

if __name__ == "__main__":
    main()
