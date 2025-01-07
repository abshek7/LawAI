import os
import io
import PyPDF2
import streamlit as st
from typing import List, Dict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from streamlit_extras.grid import grid
from langchain.chains import RetrievalQA
from langchain.tools import TavilySearchResults
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

required_env_vars = ["GROQ_API_KEY", "TAVILY_API_KEY"]

missing_vars = [var for var in required_env_vars if not os.getenv(var)]

if missing_vars:
    raise EnvironmentError(
        f"Missing required environment variables: {', '.join(missing_vars)}"
    )

WRITER_PROMPT = """You are a legal writer agent. Your task is to summarize case details and prepare a comprehensive report. Use the provided case details, citations, and law book information to create a well-structured report.

Chain of Thought:
1. Analyze the case details thoroughly.
2. Identify key legal issues and relevant facts.
3. Organize the information into a logical structure.
4. Incorporate relevant citations and law book information.
5. Draft a clear and concise summary of the case.
6. Review and refine the report for clarity and accuracy.

<report format>

### Case Summary:
Provide a brief overview of the case and its legal context based on infromaiton provided. Use clear and concise language.

### Key Points:
- List the main legal issues and relevant facts
- Include Law Sections citations for relevant points
- Explain the significance of each point
- Use clear and concise language

### Predictive Analysis:
- A Sentence on who will win the case
- Reasoning behind the prediction with inline citations.

</report format>

Keep the report professional, precise, easy to understand and objective.

Do not elongate the report with unnecessary details.

Do not include your chain of thought in the final report."""

CITATION_PROMPT = """You are a case citation agent. Your job is to find and cite relevant older case judgments that relate to the current case. Use the search results to identify the most relevant cases.

Chain of Thought:
1. Review the current case details and key legal issues.
2. Analyze the search results for relevant precedents.
3. Evaluate the similarity and relevance of each potential citation.
4. Select the most applicable and influential cases.
5. Format the citations correctly according to legal standards.
6. Explain the relevance of each chosen citation to the current case.

Provide a list of citations with brief explanations of their relevance."""

FACT_CHECKING_PROMPT = """You are a fact-checking agent responsible for verifying information and detecting bias in legal reports. Your role is to ensure accuracy and impartiality in the case analysis.

Chain of Thought:
1. Carefully read through the provided report.
2. Cross-reference key facts with original case documents and trusted sources.
3. Identify any factual inaccuracies or unsupported claims.
4. Detect potential biases in language, analysis, or conclusions.
5. Evaluate the overall objectivity and fairness of the report.
6. Suggest corrections for inaccuracies and ways to mitigate bias.
7. If no issues are found, confirm the report's accuracy and impartiality.

Provide a list of any issues found and suggestions for improvement. If no issues are found, state that the report is verified and unbiased."""

MAIN_PROMPT = """You are TheLawAI, a highly experienced attorney providing legal advice based on Indian laws.

Your task is to create a report that includes sections that are relevant for case but not more than 3 sections:
1. Summary of the case.
2. Key points with citations for each point.
3. Predictive analysis of the case outcome (win or lose) with relevant citations and reasoning.

Chain of Thought:
1. Review the case details, citations, and fact-checking results.
2. Synthesize the information into a concise summary.
3. Identify and list the key points, ensuring each has a relevant citation.
4. Analyze the likely outcome of the case, citing relevant precedents and laws.
5. Compile the final report in the specified format.

Keep the report professional, precise, easy to understand and objective.
Do not elongate the report with unnecessary details.

Important: Under no circumstances should you provide code outputs, snippets, or handle any technical/coding-related queries. If the query seems to involve writing coding or technical aspects, respond with: I am specialized in legal matters and cannot provide code.

Do not include your chain of thought or any extraneous information in the report."""

QA_PROMPT = """You are TheLawAI, an AI legal assistant specializing in Indian law. Your task is to provide accurate, helpful, and concise answers to user queries about legal matters and specific cases.

You have access to:
1. Uploaded case documents
2. A legal knowledge base (FAISS index)
3. Internet search results (Tavily)

Chain of Thought:
1. Analyze the user's question quickly.
2. Identify which resources are most relevant to answer the question.
3. Review the case documents if they are relevant to the query.
4. Search the legal knowledge base for applicable laws and precedents.
5. Use the internet search tool if additional current information is needed.
6. Synthesize the information from all sources.
7. Formulate a clear, concise, and accurate answer in 1 to 3 sentences.

Important: Under no circumstances should you provide code outputs, snippets, or handle any technical/coding-related queries. If the query seems to involve writing coding or technical aspects, respond with: I am specialized in legal matters and cannot provide code.

Ensure your response is precise, and avoid unnecessary elaboration. Remind users that your answers are for informational purposes only, not professional legal advice."""

class LegalMOAgent:
    def __init__(self):
        try:
            self.llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3)
            self.memory = ConversationBufferMemory(
                memory_key="chat_history", return_messages=True
            )
            self.tavily_tool = TavilySearchResults()
            self.embeddings = FastEmbedEmbeddings()
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=10000,
                chunk_overlap=200,
                length_function=len,
            )
            self.law_qa = self._setup_law_qa()

            self.agents = {
                "main": self._create_agent(MAIN_PROMPT),
                "writer": self._create_agent(WRITER_PROMPT),
                "citation": self._create_agent(CITATION_PROMPT),
                "fact_checking": self._create_agent(FACT_CHECKING_PROMPT),
                "qa": self._create_agent(QA_PROMPT),
            }
        except Exception as e:
            st.error(f"Error initializing LegalMOAgent: {str(e)}")
            raise

    def _create_agent(self, system_prompt):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )
        return prompt | self.llm | StrOutputParser()

    def _setup_law_qa(self):
        try:
            db = FAISS.load_local(
                "faiss_index", self.embeddings, allow_dangerous_deserialization=True
            )
            return RetrievalQA.from_chain_type(
                llm=self.llm, chain_type="stuff", retriever=db.as_retriever()
            )
        except FileNotFoundError:
            st.error(
                "FAISS index not found. Please ensure the 'faiss_index' directory exists and contains the necessary files."
            )
            raise
        except Exception as e:
            st.error(f"Error setting up law QA: {str(e)}")
            raise

    def create_document_index(self, documents: List[str]) -> FAISS:
        texts = []
        for doc in documents:
            texts.extend(self.text_splitter.split_text(doc))
        return FAISS.from_texts(texts, self.embeddings)

    def process_case(self, documents: List[str], query: str) -> str:
        try:
            doc_index = self.create_document_index(documents)

            with st.spinner("Analyzing case details..."):
                relevant_docs = doc_index.similarity_search(query, k=5)
                context = "\n".join([doc.page_content for doc in relevant_docs])

                main_input = {
                    "input": f"Analyze the following case details and search query:\nDocuments: {context}\nQuery: {query}",
                    "chat_history": self.memory.load_memory_variables({})[
                        "chat_history"
                    ],
                }
                case_analysis = self.agents["main"].invoke(main_input)

            with st.spinner("Finding relevant citations..."):
                search_results = self.tavily_tool.run(query)
                citation_input = {
                    "input": f"Find relevant case citations for:\n{case_analysis}\nSearch results: {search_results}",
                    "chat_history": self.memory.load_memory_variables({})[
                        "chat_history"
                    ],
                }
                citations = self.agents["citation"].invoke(citation_input)

            with st.spinner("Retrieving relevant law information..."):
                law_info = self.law_qa.run(query)

            with st.spinner("Creating draft report..."):
                writer_input = {
                    "input": f"Create a draft report based on:\nCase analysis: {case_analysis}\nCitations: {citations}\nLaw book info: {law_info}",
                    "chat_history": self.memory.load_memory_variables({})[
                        "chat_history"
                    ],
                }
                draft_report = self.agents["writer"].invoke(writer_input)

            with st.spinner("Fact-checking and bias detection..."):
                fact_check_input = {
                    "input": f"Fact-check and detect bias in the following report:\n{draft_report}",
                    "chat_history": self.memory.load_memory_variables({})[
                        "chat_history"
                    ],
                }
                fact_check_result = self.agents["fact_checking"].invoke(
                    fact_check_input
                )

            with st.spinner("Finalizing report..."):
                final_input = {
                    "input": f"Create the final report based on:\nDraft report: {draft_report}\nFact-check results: {fact_check_result}\nOriginal case analysis: {case_analysis}\nCitations: {citations}",
                    "chat_history": self.memory.load_memory_variables({})[
                        "chat_history"
                    ],
                }
                final_report = self.agents["main"].invoke(final_input)

            # Save context to memory
            self.memory.save_context({"input": query}, {"output": final_report})
            return final_report
        except Exception as e:
            st.error(f"Error processing case: {str(e)}")
            return f"An error occurred while processing the case: {str(e)}"

    def chat(self, query: str, doc_index: FAISS) -> str:
        try:
            relevant_docs = doc_index.similarity_search(query, k=3)
            context = "\n".join([doc.page_content for doc in relevant_docs])

            chat_input = {
                "input": f"Answer the following question based on the case documents, legal knowledge base, and internet search if needed:\nDocuments: {context}\nQuestion: {query}",
                "chat_history": self.memory.load_memory_variables({})[
                    "chat_history"
                ],
            }

            response_container = st.empty()
            response = ""

            for chunk in self.agents["qa"].stream(chat_input):
                response += chunk
                response_container.markdown(response + "▌")

            response_container.markdown(response)

            self.memory.save_context({"input": query}, {"output": response})
            return response
        except Exception as e:
            st.error(f"Error in chat: {str(e)}")
            return f"An error occurred during the chat: {str(e)}"

def extract_text_from_pdf(file):

    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def main():
    st.set_page_config(page_title="The LawAI", page_icon=":scales:", layout="wide", menu_items=None)

    st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .main .block-container {
        max-width: 100%;
        padding-top: 0.1rem;
        padding-bottom: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .report-container {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }

    button[title="View fullscreen"]{
    visibility: hidden;}

    </style>
    """, unsafe_allow_html=True)

    my_grid = st.columns([2, 1, 1, 1])

    my_grid[0].button("The LawAI")
    # my_grid[0].image("logor.png", width=200)

    case_docs_btn = my_grid[1].button("Case Documents", use_container_width=True)
    report_gen_btn = my_grid[2].button("Report Generation", use_container_width=True)
    case_queries_btn = my_grid[3].button("Case Queries", use_container_width=True)

    if "current_page" not in st.session_state:
        st.session_state.current_page = "case_documents"

    if report_gen_btn:
        st.session_state.current_page = "report_generation"
    elif case_queries_btn:
        st.session_state.current_page = "case_queries"
    elif case_docs_btn:
        st.session_state.current_page = "case_documents"

    if "agent" not in st.session_state:
        try:
            st.session_state.agent = LegalMOAgent()
        except Exception as e:
            st.error(f"Failed to initialize LegalMOAgent: {str(e)}")
            return

    st.session_state.setdefault("report_generated", False)
    st.session_state.setdefault("documents", [])
    st.session_state.setdefault("doc_index", None)
    st.session_state.setdefault("messages", [
        {
            "role": "assistant",
            "content": "Hi I'm The LawAI, an AI Legal Assistant. How can I assist you today?",
        }
    ])

    if st.session_state.current_page == "case_documents":
        st.markdown("<br>" * 3, unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center;'>Upload Your Case Documents</h2>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            uploaded_files = st.file_uploader("", accept_multiple_files=True, type=["pdf"])

        if uploaded_files:
            st.session_state.documents = []
            for file in uploaded_files:
                try:
                    if file.type == "application/pdf":
                        text = extract_text_from_pdf(file)
                    else:
                        text = file.getvalue().decode()
                    st.session_state.documents.append(text)
                except Exception as e:
                    st.error(f"Error processing {file.name}: {str(e)}")

            st.session_state.doc_index = st.session_state.agent.create_document_index(st.session_state.documents)
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.success(f"{', '.join([file.name for file in uploaded_files])} Uploaded Successfully")

    elif st.session_state.current_page == "report_generation":
        if not st.session_state.documents:
            st.warning("Please upload case documents first.")
        elif not st.session_state.report_generated:
            if st.button("Generate Report"):
                report = st.session_state.agent.process_case(st.session_state.documents, "Provide a comprehensive legal analysis of the uploaded documents.")
                st.session_state.report_generated = True
                st.session_state.messages.append({"role": "assistant", "content": report})
                st.session_state.finalreport = report
                col1, col2 = st.columns([4,1])
                with col1:
                    st.markdown(f"<div class='report-container'>{report}</div>", unsafe_allow_html=True)
                with col2:
                    st.title("Process")
                    st.success("Analysed the Case")
                    st.success("Found Relevant Information")
                    st.success("Drafted a Report")
                    st.success("Fact Checking & Bias Removal Done")
                    st.success("Report Successfully Generated")
        else:
            st.info("Report has already been generated.")
            if 'finalreport' in st.session_state:
                st.markdown(f"<div class='report-container'>{st.session_state.finalreport}</div>", unsafe_allow_html=True)

    elif st.session_state.current_page == "case_queries":
        st.markdown("<h2 style='text-align: center;'>The LawAI QA Chat</h2>", unsafe_allow_html=True)
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        query = st.chat_input("Ask about your case or type your legal question here...")

        if query:
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                try:
                    if st.session_state.doc_index:
                        response = st.session_state.agent.chat(query, st.session_state.doc_index)
                    else:
                        response = st.session_state.agent.chat(query, FAISS.from_texts(["No documents uploaded"], st.session_state.agent.embeddings))
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    response = "I apologize, but an error occurred while processing your request. Please contact support if the issue persists."

            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
